"""
ECFFT for Group BaseFold
=========================

Pure-Python implementation of the ECFFT machinery needed for the group-valued
BaseFold protocol from `eprint 2025/1325 <https://eprint.iacr.org/2025/1325>`_
(Eagen & Gabizon, "Revisiting the IPA-sumcheck connection").

References:

  * **ECFFT Part I**:  https://arxiv.org/pdf/2107.08473.pdf
  * **ECFFT Part II**: https://www.math.toronto.edu/swastik/ECFFT2.pdf
  * **BaseFold**:      https://eprint.iacr.org/2025/1325

We work over the BN-254 base field  Fq,  q given below.


Overview
--------

The standard FFT requires roots of unity of large 2-power order.  The BN-254
base field has no such roots.  The ECFFT replaces the squaring map  x ↦ x²
with a rational map  ψ(x) = (x − b)²/x  induced by a good isogeny on an
elliptic curve.  ψ is 2-to-1 on evaluation domains built from curve points,
giving the same recursive halving that powers the FFT.

This file provides:

  1. **Field arithmetic** over Fq  (§1)
  2. **Elliptic curve** infrastructure: good curves, points, isogenies  (§2–§3)
  3. **FRI domain construction**: the layer sequence  L₀ → L₁ → ⋯ → Lₖ  (§4)
  4. **ECFFT Part II pointwise FRI fold**: the correct fold for protocols  (§5)
  5. **Group-valued BaseFold**: FRI folding over curve points  (§6)

For the general-purpose FFTree (ENTER/EXIT/EXTEND/DEGREE) and the global
Part I decomposition, see ``ecfft_fftree.py``.


§1  Field Arithmetic
---------------------
"""

# ═══════════════════════════════════════════════════════════════════════════
# §1  Fq — the BN-254 base field
# ═══════════════════════════════════════════════════════════════════════════

q = 21888242871839275222246405745257275088696311157297823662689037894645226208583

def fadd(a, b):  return (a + b) % q
def fsub(a, b):  return (a - b) % q
def fmul(a, b):  return (a * b) % q
def fneg(a):     return (q - a) % q
def finv(a):
    assert a % q != 0, "division by zero"
    return pow(a, q - 2, q)
def fdiv(a, b):  return fmul(a, finv(b))
def fpow(a, n):  return pow(a % q, int(n) % (q - 1), q)
def fsqrt(a):
    """Square root in Fq, or None if `a` is a QNR.  (q ≡ 3 mod 4)"""
    a = a % q
    if a == 0: return 0
    r = pow(a, (q + 1) // 4, q)
    return r if fmul(r, r) == a else None

def batch_inv(xs):
    """Montgomery batch inversion — one inversion, O(n) muls."""
    n = len(xs)
    if n == 0: return []
    prefix = [0] * n
    prefix[0] = xs[0]
    for i in range(1, n):
        prefix[i] = fmul(prefix[i - 1], xs[i])
    inv_acc = finv(prefix[-1])
    out = [0] * n
    for i in range(n - 1, 0, -1):
        out[i] = fmul(inv_acc, prefix[i - 1])
        inv_acc = fmul(inv_acc, xs[i])
    out[0] = inv_acc
    return out


def poly_eval(coeffs, x):
    """Horner evaluation of a polynomial given as [c0, c1, …]."""
    r = 0
    for c in reversed(coeffs):
        r = fadd(fmul(r, x), c)
    return r


# ═══════════════════════════════════════════════════════════════════════════
# §2  Good Curves,  Points,  Isogenies
# ═══════════════════════════════════════════════════════════════════════════

class GoodCurve:
    """
    E_{a,B}:  y² = x³ + a·x² + B·x,   B = b²   (ECFFT II §3).

    Distinguished 2-torsion point  T = (0, 0) ∈ ker(φ).
    """
    def __init__(self, a, bb):
        self.a  = a % q
        self.bb = bb % q
        b = fsqrt(bb)
        assert b is not None, f"bb={bb} is not a QR"
        self.b = b
        assert fsub(fmul(a, a), fmul(4, bb)) % q != 0, "singular"
        assert fsqrt(fadd(a, fmul(2, b))) is not None, "a+2b must be QR"

    def contains(self, x, y):
        lhs = fmul(y, y)
        rhs = fadd(fadd(fmul(x, fmul(x, x)), fmul(self.a, fmul(x, x))), fmul(self.bb, x))
        return lhs == rhs

    def __repr__(self):
        return f"GoodCurve(a={self.a}, B={self.bb})"


class Point:
    """Affine point on a Good Curve, or the point at infinity (curve=None)."""
    def __init__(self, x, y, curve):
        self.x = x % q if curve else 0
        self.y = y % q if curve else 0
        self.curve = curve

    @staticmethod
    def infinity(): return Point(0, 0, None)
    def is_infinity(self): return self.curve is None

    def __eq__(self, o):
        if self.is_infinity() and o.is_infinity(): return True
        if self.is_infinity() or o.is_infinity(): return False
        return self.x == o.x and self.y == o.y

    def __neg__(self):
        return self if self.is_infinity() else Point(self.x, fneg(self.y), self.curve)

    def __repr__(self):
        return "O" if self.is_infinity() else f"({self.x}, {self.y})"

    def __add__(self, o):
        if self.is_infinity(): return o
        if o.is_infinity():    return self
        a2, a4 = self.curve.a, self.curve.bb
        x1, y1, x2, y2 = self.x, self.y, o.x, o.y
        if x1 == x2:
            if fadd(y1, y2) == 0: return Point.infinity()
            num = fadd(fadd(fmul(3, fmul(x1, x1)), fmul(2, fmul(a2, x1))), a4)
            den = fmul(2, y1)
            lam = fdiv(num, den)
            nu  = fdiv(fadd(fneg(fmul(x1, fmul(x1, x1))), fmul(a4, x1)), den)
        else:
            lam = fdiv(fsub(y2, y1), fsub(x2, x1))
            nu  = fdiv(fsub(fmul(y1, x2), fmul(y2, x1)), fsub(x2, x1))
        x3 = fsub(fsub(fsub(fmul(lam, lam), a2), x1), x2)
        y3 = fsub(fneg(fmul(lam, x3)), nu)
        return Point(x3, y3, self.curve)

    def double(self):       return self + self
    def scalar_mul(self, n):
        n = int(n)
        if n < 0: return (-self).scalar_mul(-n)
        R, A = Point.infinity(), self
        while n:
            if n & 1: R = R + A
            A = A + A; n >>= 1
        return R


# ═══════════════════════════════════════════════════════════════════════════
# §3  Rational maps and the good isogeny
# ═══════════════════════════════════════════════════════════════════════════

class RationalMap:
    """P(x)/Q(x), polynomials stored as coefficient lists [c0, c1, …]."""
    def __init__(self, num, den):
        self.num = [c % q for c in num]
        self.den = [c % q for c in den]
    def __call__(self, x):
        x = x % q
        n = poly_eval(self.num, x)
        d = poly_eval(self.den, x)
        return None if d == 0 else fdiv(n, d)
    def __repr__(self):
        return f"RationalMap({self.num}/{self.den})"


def good_isogeny(curve):
    """
    Good isogeny  φ: E → E'  for odd-char Good Curve.

        r(x) = (x − b)²/x       (x-coordinate map = ψ)
        h(x) = (x² − b²)/x²     (y-scaling)
        a' = a + 6b,   B' = 4ab + 8b²
    """
    a, b, bb = curve.a, curve.b, curve.bb
    r = RationalMap([bb, fneg(fmul(2, b)), 1], [0, 1])
    h = RationalMap([fneg(bb), 0, 1], [0, 0, 1])
    codomain = GoodCurve(fadd(a, fmul(6, b)),
                         fadd(fmul(4, fmul(a, b)), fmul(8, fmul(b, b))))
    return r, h, codomain


def apply_isogeny(r, h, codomain, pt):
    if pt.is_infinity(): return Point.infinity()
    rx = r(pt.x)
    if rx is None: return Point.infinity()
    hx = h(pt.x)
    if hx is None: return Point.infinity()
    return Point(rx, fmul(hx, pt.y), codomain)


def build_isogeny_chain(gen, k):
    """Build k good isogenies from a generator of order 2^k."""
    psis, hs, curves = [], [], [gen.curve]
    g = gen
    for _ in range(k):
        r, h, cod = good_isogeny(g.curve)
        psis.append(r); hs.append(h); curves.append(cod)
        g = apply_isogeny(r, h, cod, g)
    return psis, curves, hs


# ═══════════════════════════════════════════════════════════════════════════
# §4  FRI domain construction
# ═══════════════════════════════════════════════════════════════════════════
#
# The FRI protocol operates on a sequence of evaluation domains:
#
#   L₀  →  L₁  →  ⋯  →  Lₖ
#       ψ₀     ψ₁         ψ_{k-1}
#
# Each ψᵢ is 2-to-1 on Lᵢ, halving the domain at each step.
# The pairing invariant:  ψᵢ(Lᵢ[j]) = ψᵢ(Lᵢ[j + m/2]) = Lᵢ₊₁[j].
# ═══════════════════════════════════════════════════════════════════════════


def build_fri_domains(params, log_n):
    """
    Build the FRI layer domains  L₀, …, Lₖ  from curve parameters.

    Parameters
    ----------
    params : dict
        Curve parameters with keys 'a', 'bb', 'gx', 'gy', 'k'.
    log_n : int
        Log₂ of the initial domain size.

    Returns
    -------
    layers : list of list of int
        layers[i] has size 2^{log_n - i}.
    rational_maps : list of RationalMap
        rational_maps[i] maps Lᵢ → Lᵢ₊₁.
    """
    assert log_n <= params['k']
    n = 1 << log_n

    curve = GoodCurve(params['a'], params['bb'])
    gen = Point(params['gx'], params['gy'], curve)
    scaled_gen = gen.scalar_mul(1 << (params['k'] - log_n))

    # Build isogeny chain → rational maps
    psis, curves, hs = build_isogeny_chain(scaled_gen, log_n)

    # Initial domain: x-coordinates of coset {2G + i·scaled_gen}
    coset = gen.double()
    L0, acc = [], Point.infinity()
    for _ in range(n):
        L0.append((coset + acc).x)
        acc = acc + scaled_gen

    # Build successive layers by applying ψ
    layers = [L0]
    current = L0
    for i in range(log_n):
        psi = psis[i]
        m = len(current)
        half = m // 2
        next_layer = [psi(current[j]) for j in range(half)]
        # Verify the 2-to-1 pairing
        for j in range(half):
            img = psi(current[j + half])
            assert next_layer[j] == img, \
                f"ψ pairing broken at layer {i}, j={j}: {next_layer[j]} ≠ {img}"
        layers.append(next_layer)
        current = next_layer

    return layers, psis


# ═══════════════════════════════════════════════════════════════════════════
# §5  ECFFT Part II pointwise FRI fold
# ═══════════════════════════════════════════════════════════════════════════
#
# The FRI algebraic hash  H_z  from ECFFT Part II (BSCKL22, Appendix B.2).
#
# For a degree-2 rational map  ψ(x) = u(x)/v(x)  with denominator v(x) = x,
# any polynomial P of degree < d decomposes as:
#
#   P(x) = (P₀(ψ(x)) + x · P₁(ψ(x))) · v(x)^{d/2 − 1}
#
# with deg(P₀), deg(P₁) < d/2.  For a pair (s₀, s₁) with ψ(s₀) = ψ(s₁),
# setting e = d/2 − 1:
#
#   P(s₀) / s₀ᵉ = P₀(t) + s₀ · P₁(t)
#   P(s₁) / s₁ᵉ = P₀(t) + s₁ · P₁(t)
#
# Solving this 2×2 system and evaluating P₀ + z·P₁ at t gives:
#
#   H_z[P](t)  =  a + slope · (z − s₀)
#
# where  a = P(s₀)/s₀ᵉ,  b = P(s₁)/s₁ᵉ,  slope = (b − a)/(s₁ − s₀).
#
# This is POINTWISE: each output depends on exactly 2 inputs.  This is what
# makes FRI verification O(1) per query.
# ═══════════════════════════════════════════════════════════════════════════


def ecfri_fold_step(word, layers, round_idx, degree_bound, z):
    """
    One round of pointwise FRI folding.

    Parameters
    ----------
    word : list of int
        Evaluations on layer Lᵢ (size m = |Lᵢ|).
    layers : list of list of int
        FRI domain layers from build_fri_domains().
    round_idx : int
        Current round (0-based).  Uses layers[round_idx].
    degree_bound : int
        Current degree bound dᵢ.  Must be even, ≤ m.
    z : int
        Verifier challenge.

    Returns
    -------
    out : list of int
        Evaluations on layer Lᵢ₊₁ (size m/2).
    """
    layer = layers[round_idx]
    m = len(layer)
    assert len(word) == m
    assert degree_bound % 2 == 0
    assert degree_bound <= m
    half = m // 2
    e = degree_bound // 2 - 1

    # Batch-invert the pair differences for efficiency
    diffs = [fsub(layer[j + half], layer[j]) for j in range(half)]
    diff_invs = batch_inv(diffs)

    out = [0] * half
    for j in range(half):
        s0 = layer[j]
        s1 = layer[j + half]

        # Normalize by v(s)^e = s^e
        if e == 0:
            a = word[j]
            b = word[j + half]
        else:
            a = fdiv(word[j], fpow(s0, e))
            b = fdiv(word[j + half], fpow(s1, e))

        # Evaluate line through (s0, a) and (s1, b) at z
        slope = fmul(fsub(b, a), diff_invs[j])
        out[j] = fadd(a, fmul(slope, fsub(z, s0)))

    return out


def ecfri_fold(word, layers, degree_bound, challenges):
    """
    Multi-round pointwise FRI fold.

    Parameters
    ----------
    word : list of int
        Evaluations on layers[0].
    layers : list of list of int
        FRI domain layers from build_fri_domains().
    degree_bound : int
        Initial degree bound (halved each round).
    challenges : list of int
        One challenge per round.

    Returns
    -------
    folded : list of int
        Evaluations on layers[len(challenges)].
    """
    current = list(word)
    d = degree_bound
    for i, z in enumerate(challenges):
        current = ecfri_fold_step(current, layers, i, d, z)
        d = d // 2
    return current


def ecfri_verify_query(layers, round_idx, degree_bound, j, f_s0, f_s1, z):
    """
    Verify a single FRI fold query in O(1).

    Given f(s₀) and f(s₁) for pair index j, returns the expected fold value
    at layers[round_idx + 1][j].

    Parameters
    ----------
    layers : list of list of int
    round_idx, degree_bound : int
    j : int
        Pair index (0 ≤ j < |Lᵢ|/2).
    f_s0, f_s1 : int
        The two opened evaluations.
    z : int
        Verifier challenge for this round.

    Returns
    -------
    expected : int
        Expected fold value at layers[round_idx + 1][j].
    """
    layer = layers[round_idx]
    m = len(layer)
    half = m // 2
    e = degree_bound // 2 - 1

    s0 = layer[j]
    s1 = layer[j + half]
    diff_inv = finv(fsub(s1, s0))

    if e == 0:
        a, b = f_s0, f_s1
    else:
        a = fdiv(f_s0, fpow(s0, e))
        b = fdiv(f_s1, fpow(s1, e))

    slope = fmul(fsub(b, a), diff_inv)
    return fadd(a, fmul(slope, fsub(z, s0)))


# ═══════════════════════════════════════════════════════════════════════════
# §6  Group-valued BaseFold  (eprint 2025/1325, Section 7)
# ═══════════════════════════════════════════════════════════════════════════
#
# In IPA verification, the "decide" step computes:
#
#   G(r) = Σᵢ sᵢ · Gᵢ
#
# where G₀,…,G_{n-1} are SRS generators and sᵢ are IPA-challenge-derived
# scalars.  This O(n) MSM dominates recursive verification.
#
# Group BaseFold replaces this with a FRI-like protocol over GROUP ELEMENTS:
#
# 1. SRS ENCODING (one-time precompute):
#    g₀[j] = Σᵢ L₀[j]ⁱ · Gᵢ   for each evaluation point in L₀.
#
# 2. FRI ROUNDS (k = log n rounds):
#    For round i:
#      a. Prover commits gᵢ  (Merkle root of group elements on Lᵢ)
#      b. Verifier sends challenge zᵢ
#      c. Prover folds:  gᵢ₊₁ = ECFFT2-fold(gᵢ, zᵢ)  — pointwise over
#         group elements, 4 scalar muls per pair
#
# 3. FINAL CHECK:
#    gₖ is a single group element; verifier checks it matches.
#
# 4. QUERY PHASE (~43 queries for 128-bit security):
#    For each query, verify fold consistency across all k rounds:
#
#      e = dᵢ/2 − 1
#      a = gᵢ[j] · (1/s₀ᵉ)            (scalar mul)
#      b = gᵢ[j + m/2] · (1/s₁ᵉ)      (scalar mul)
#      slope = (b − a) · diff_inv       (scalar mul)
#      expected = a + slope · (z − s₀)  (scalar mul)
#      CHECK: expected == gᵢ₊₁[j']     (group element equality)
#
#    Total per round per query: 4 scalar muls + Merkle path verification.
#
# Group elements are (x, y) pairs on Grumpkin (BN254 G1 over Fq).
# For Merkle commitments, each leaf is hash(x, y) using Poseidon2.
# ═══════════════════════════════════════════════════════════════════════════


def basefold_group_fold_step(g_word, layers, round_idx, degree_bound, z):
    """
    BaseFold prover: fold a group-element vector using the ECFFT2 hash.

    Same formula as ecfri_fold_step, but over (x, y) curve points.
    "Division" by a scalar → multiplication by its inverse;
    "addition" → group addition.

    Parameters
    ----------
    g_word : list of (int, int) or None
        Group elements on layer Lᵢ (size m).  None = point at infinity.
    layers : list of list of int
        FRI domain layers from build_fri_domains().
    round_idx : int
    degree_bound : int
    z : int
        Verifier challenge.

    Returns
    -------
    g_out : list of (int, int) or None
        Group elements on layer Lᵢ₊₁ (size m/2).
    """
    layer = layers[round_idx]
    m = len(layer)
    assert len(g_word) == m
    assert degree_bound % 2 == 0
    half = m // 2
    e = degree_bound // 2 - 1

    # Batch-invert pair differences
    diffs = [fsub(layer[j + half], layer[j]) for j in range(half)]
    diff_invs = batch_inv(diffs)

    # Batch-invert s^e values for normalization
    if e > 0:
        s_e_vals = [fpow(layer[j], e) for j in range(m)]
        s_e_invs = batch_inv(s_e_vals)
    else:
        s_e_invs = [1] * m

    g_out = [None] * half
    for j in range(half):
        s0_e_inv = s_e_invs[j]
        s1_e_inv = s_e_invs[j + half]

        # a = g_word[j] · (1/s₀ᵉ)
        a = _group_scalar_mul(g_word[j], s0_e_inv)
        # b = g_word[j + half] · (1/s₁ᵉ)
        b = _group_scalar_mul(g_word[j + half], s1_e_inv)

        # slope = (b − a) · diff_inv
        b_minus_a = _group_add(b, _group_neg(a))
        slope = _group_scalar_mul(b_minus_a, diff_invs[j])

        # out = a + slope · (z − s₀)
        z_minus_s0 = fsub(z, layer[j])
        g_out[j] = _group_add(a, _group_scalar_mul(slope, z_minus_s0))

    return g_out


def basefold_verify_query(layers, round_idx, degree_bound, j, g_s0, g_s1, z):
    """
    BaseFold verifier: check a single fold query over group elements.

    Returns the expected fold value as a group element.  The verifier
    compares this against the committed value at layers[round_idx + 1][j].

    Parameters
    ----------
    layers : list of list of int
    round_idx, degree_bound, j, z : as in ecfri_verify_query
    g_s0, g_s1 : (int, int) or None
        Opened group elements at the pair.

    Returns
    -------
    expected : (int, int) or None
        Expected group element at layers[round_idx + 1][j].
    """
    layer = layers[round_idx]
    m = len(layer)
    half = m // 2
    e = degree_bound // 2 - 1

    s0 = layer[j]
    s1 = layer[j + half]
    diff_inv = finv(fsub(s1, s0))

    if e == 0:
        a, b = g_s0, g_s1
    else:
        a = _group_scalar_mul(g_s0, finv(fpow(s0, e)))
        b = _group_scalar_mul(g_s1, finv(fpow(s1, e)))

    b_minus_a = _group_add(b, _group_neg(a))
    slope = _group_scalar_mul(b_minus_a, diff_inv)
    z_minus_s0 = fsub(z, s0)
    return _group_add(a, _group_scalar_mul(slope, z_minus_s0))


# ── Group element helpers (affine arithmetic over Fq) ──
# Points are (x, y) tuples or None for the identity.

def _group_add(p, q_pt):
    """Add two affine points on a short Weierstrass curve (a=0, i.e. Grumpkin)."""
    if p is None:
        return q_pt
    if q_pt is None:
        return p
    px, py = p
    qx, qy = q_pt
    if px == qx:
        if py == qy and py != 0:
            # Point doubling:  λ = 3x²/(2y)  for y² = x³ + b  (a = 0)
            lam = fdiv(fmul(3, fmul(px, px)), fmul(2, py))
        else:
            return None  # point at infinity
    else:
        lam = fdiv(fsub(qy, py), fsub(qx, px))
    rx = fsub(fsub(fmul(lam, lam), px), qx)
    ry = fsub(fmul(lam, fsub(px, rx)), py)
    return (rx, ry)


def _group_neg(p):
    """Negate an affine point."""
    if p is None:
        return None
    return (p[0], fneg(p[1]))


def _group_scalar_mul(p, scalar):
    """Scalar multiplication by double-and-add."""
    if p is None:
        return None
    scalar = scalar % q
    if scalar == 0:
        return None
    if scalar == 1:
        return p
    result = None
    base = p
    while scalar > 0:
        if scalar & 1:
            result = _group_add(result, base)
        base = _group_add(base, base)
        scalar >>= 1
    return result


if __name__ == "__main__":
    print("ECFFT for Group BaseFold — core algorithms")
    print()
    print("  from ecfft_algorithms import build_fri_domains, ecfri_fold_step")
    print("  from ecfft_algorithms import basefold_group_fold_step, basefold_verify_query")
    print("  from ecfft_params_2_20 import params")
    print()
    print("For the general FFTree (ENTER/EXIT/EXTEND/DEGREE), see ecfft_fftree.py")
