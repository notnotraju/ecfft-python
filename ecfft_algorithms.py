"""
ECFFT: Elliptic Curve Fast Fourier Transform — Algorithms
==========================================================

A pure-Python implementation of the ECFFT, based on the Rust implementation
at https://github.com/andrewmilson/ecfft.

References:

  * **ECFFT Part I**:  https://arxiv.org/pdf/2107.08473.pdf
  * **ECFFT Part II**: https://www.math.toronto.edu/swastik/ECFFT2.pdf

We work over the BN-254 base field  Fq,  q given below.


The FFT analogy — even/odd decomposition
------------------------------------------

In the classic radix-2 FFT over a domain D = {ω⁰, ω¹, …, ω^{n-1}} where ω
is a primitive n-th root of unity, the key decomposition is::

    f(x) = f_even(x²) + x · f_odd(x²)

The squaring map  x ↦ x²  is 2-to-1 on D:  it sends ω^i and ω^{i+n/2} to the
same point (ω²)^i.  This halves the domain, and we recurse.

The ECFFT replaces the squaring map with the rational map  ψ(x) = (x−b)²/x
induced by a good isogeny on an elliptic curve.  ψ is also 2-to-1 on the
evaluation domain.  The decomposition becomes::

    f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x))     [ENTER]

where  u  holds the low coefficients (indices 0..n/2)  and  v  holds the high
coefficients (indices n/2..n).  This is the split in ``_enter_impl``.

The inverse (``_exit_impl``) recovers u, v from evaluations via modular
reduction:  u = f mod x^{n/2},  then  v = (f − u) / x^{n/2}.

The ``_extend_impl`` operation (Lemma 3.2 of ECFFT I) plays the role of the
"twiddle factor" in the classic FFT: given evaluations of a polynomial on one
half of the domain (moiety S₀), it computes the evaluations on the other half
(moiety S₁) in O(n log n) via a matrix-based decompose/recurse/recombine
strategy.


Pointwise folding  (FRI-like decomposition)
---------------------------------------------

The key operation for FRI-like protocols: given f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x))
evaluated on an n-point domain, decompose into evaluations of u and v on the
half-size ψ-image domain, then combine with a random challenge α::

    f_folded(y) = u(y) + α · v(y)      (degree < n/2, on n/2 points)

Unlike the classic FFT where folding is a pointwise 2×2 solve (because
f(ω^i) and f(−ω^i) share the same u, v values), the ECFFT decomposition
requires the FFTree's **modular reduction** machinery.  The reason is that
the ECFFT's ENTER recombines u and v via the extend operation, which is a
non-local algebraic operation — not a simple per-pair formula.

The correct decomposition (one level of EXIT) is::

    u_on_S0  = modular_reduce(⟨f ≀ S⟩,  ⟨x^{n/2} ≀ S⟩,  ⟨Z₀² mod x^{n/2} ≀ S⟩)[S₀]
    v_on_S0  = (f(S₀) − u(S₀)) / S₀^{n/2}

Available as:

  * ``ecfft_decompose_step(evals, tree)`` → (u_evals, v_evals) on the subtree domain
  * ``ecfft_fold_step(evals, tree, alpha)`` → evaluations of u + α·v
  * ``ecfft_fold(evals, tree, [α₁, α₂, …])`` → multi-round folding

See §11 at the bottom of this file.


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
# §4  BinaryTree — heap-layout binary tree,  mirrors src/utils.rs
# ═══════════════════════════════════════════════════════════════════════════

class BinaryTree:
    """
    Flat array of length 2n storing a complete binary tree with n leaves::

             a            ← layer d-1 = root    (data[1])
            / \\
           b   c          ← layer d-2           (data[2..4])
          / \\ / \\
         w  x y  z        ← layer 0   = leaves  (data[n..2n])

    Layout: ``data = [unused, root, layer_{d-2}, …, leaves]``.
    ``leaves = data[n:]``.  ``get_layer(i)`` has  n >> i  elements.
    """
    def __init__(self, data):
        assert len(data) > 0 and (len(data) & (len(data) - 1)) == 0
        self.data = list(data)

    @property
    def n(self):
        """Number of leaves."""
        return len(self.data) // 2

    def leaves(self):        return self.data[self.n:]
    def num_layers(self):    return self.n.bit_length()   # = log2(2n)

    def get_layer(self, i):
        """Layer i: layer 0 = leaves (size n), layer 1 = size n/2, …"""
        sz = self.n >> i
        return self.data[sz : 2 * sz]

    def set_layer(self, i, vals):
        sz = self.n >> i
        self.data[sz : 2 * sz] = vals

    def get_layers(self):
        """Return [layer_0 (leaves), layer_1, …, layer_{d-1} (root)].

        Total number of layers = log₂(n) + 1  (for n a power of 2).
        """
        log_n = (self.n - 1).bit_length() if self.n > 1 else 0
        return [self.get_layer(i) for i in range(log_n + 1)]


# ═══════════════════════════════════════════════════════════════════════════
# §5  Mat2x2 — 2×2 matrices over Fq
# ═══════════════════════════════════════════════════════════════════════════

class Mat2x2:
    __slots__ = ('a', 'b', 'c', 'd')
    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a % q, b % q, c % q, d % q
    @staticmethod
    def identity():
        return Mat2x2(1, 0, 0, 1)
    def det(self):
        return fsub(fmul(self.a, self.d), fmul(self.b, self.c))
    def inv(self):
        di = finv(self.det())
        return Mat2x2(fmul(self.d, di), fmul(fneg(self.b), di),
                       fmul(fneg(self.c), di), fmul(self.a, di))
    def __mul__(self, v):
        """Mat2x2 * (v0, v1) → (w0, w1)"""
        return (fadd(fmul(self.a, v[0]), fmul(self.b, v[1])),
                fadd(fmul(self.c, v[0]), fmul(self.d, v[1])))
    def __repr__(self):
        return f"[[{self.a},{self.b}],[{self.c},{self.d}]]"


# ═══════════════════════════════════════════════════════════════════════════
# §6  FFTree — the core data structure,  mirrors src/fftree.rs
# ═══════════════════════════════════════════════════════════════════════════

class FFTree:
    """
    The ECFFT "FFTree" — a precomputed structure over an evaluation domain S
    of size n (a power of 2).

    **Domain layout (leaf order):**

    Leaves are the x-coordinates  [x(O), x(O+G), x(O+2G), …, x(O+(n-1)G)]
    laid out sequentially.

    **Moiety pairing:**

    At each tree level, the first half pairs with the second half under the
    corresponding ψ map.  At the leaf level::

        ψ₀(leaf[i]) = ψ₀(leaf[i + n/2])     for i = 0, …, n/2 − 1

    This holds because  (n/2)·G = T = (0,0)  is the kernel of ψ₀.

    **Subtree:**

    The subtree is formed by taking the *even-indexed* elements from every
    layer (``step_by(2)``), giving a tree of half the size over the ψ-image
    domain.  It uses ``rational_maps[:-1]`` (all maps except the last).

    **Naming conventions (matching the Rust):**

    =========== ============================================================
    ``f``       BinaryTree of domain x-coordinates
    ``xnn_s``   evaluation table  ⟨X^{n/2} ≀ S⟩  (S = all leaves)
    ``z0_s1``   ⟨Z₀ ≀ S₁⟩  — vanishing poly of S₀ evaluated on S₁
    ``z1_s0``   ⟨Z₁ ≀ S₀⟩
    =========== ============================================================

    S₀ = even-indexed leaves,  S₁ = odd-indexed leaves.
    """

    def __init__(self, leaves, rational_maps):
        """
        Build an FFTree from raw leaves and the chain of ψ rational maps.
        This mirrors ``FFTree::new`` in fftree.rs.
        """
        n = len(leaves)
        assert n > 0 and (n & (n - 1)) == 0
        log_n = n.bit_length() - 1
        assert log_n == len(rational_maps)

        # --- Build the BinaryTree of domain x-coords ---
        data = [0] * (2 * n)
        data[n:] = list(leaves)
        self.f = BinaryTree(data)

        # Generate internal nodes by applying rational maps layer by layer.
        # Layer 0 = leaves.  Layer 1 is built from layer 0 via rational_maps[0], etc.
        for k, rmap in enumerate(rational_maps):
            prev = self.f.get_layer(k)          # size n >> k
            layer_sz = len(prev) // 2           # size of the layer we're building
            layer = [0] * layer_sz
            for i in range(layer_sz):
                layer[i] = rmap(prev[i])
                # Assertion: rmap(prev[i]) == rmap(prev[i + layer_sz])
                assert layer[i] == rmap(prev[i + layer_sz]), \
                    f"ψ pairing broken at level {k}, index {i}"
            self.f.set_layer(k + 1, layer)

        self.rational_maps = list(rational_maps)

        # --- Delegate to _from_tree for all the precomputation ---
        self._from_tree()

    def _from_tree(self):
        """
        Precompute everything needed for ENTER / EXIT / EXTEND / etc.
        Mirrors ``FFTree::from_tree`` in fftree.rs.
        """
        n = self.f.n                       # number of leaves
        log_n = n.bit_length() - 1
        s = self.f.leaves()                # layer 0

        # --- subtree (recursive) ---
        self.subtree = self._derive_subtree()

        # --- xnn: ⟨X^{n/2} ≀ S⟩ and its inverse ---
        nn = n // 2
        self.xnn_s     = [fpow(x, nn) for x in s]
        self.xnn_s_inv = batch_inv(list(self.xnn_s))

        # --- S₀, S₁ (even / odd indexed leaves) ---
        s0 = s[0::2]
        s1 = s[1::2]

        # --- Decomposition / recombination matrices (Lemma 3.2, ECFFT I) ---
        # One BinaryTree for recombine, one for decompose, each with n entries.
        # We store them as flat lists-of-Mat2x2, indexed like BinaryTree layers.
        # Layer k has n >> (k+1) matrices.  (Same indexing as f layers.)
        layers = self.f.get_layers()       # layers[0] = leaves, etc.
        self.recombine = []   # list of lists of Mat2x2, one per tree layer
        self.decompose = []

        for k in range(log_n):
            lyr = layers[k]                 # domain values at this level
            d = len(lyr) // 2               # = n >> (k+1)
            if d <= 1:
                self.recombine.append([])
                self.decompose.append([])
                continue
            rmap = self.rational_maps[k]
            v_poly = rmap.den               # denominator polynomial
            exp = d // 2 - 1
            if exp < 0: exp = 0
            rmats = []
            dmats = []
            for i in range(d):              # d matrices, pairing l[i] with l[i+d]
                s0_val = lyr[i]
                s1_val = lyr[i + d]
                v0 = fpow(poly_eval(v_poly, s0_val), exp)
                v1 = fpow(poly_eval(v_poly, s1_val), exp)
                m = Mat2x2(v0, fmul(s0_val, v0), v1, fmul(s1_val, v1))
                rmats.append(m)
                dmats.append(m.inv())
            self.recombine.append(rmats)
            self.decompose.append(dmats)

        # --- Vanishing polynomial evaluation tables ---
        self.z0_s1 = []
        self.z1_s0 = []
        if n > 2:
            st = self.subtree
            # z0_s1 from subtree's vanishing data (O(n log n))
            st_z0_s0 = []
            st_z1_s0 = []
            for y in st.z0_s1:
                st_z0_s0.extend([0, y])
            for y in st.z1_s0:
                st_z1_s0.extend([y, 0])
            st_z0_s1 = self._extend_impl(st_z0_s0, 'S1')
            st_z1_s1 = self._extend_impl(st_z1_s0, 'S1')
            self.z0_s1 = [fmul(a, b) for a, b in zip(st_z0_s1, st_z1_s1)]
            # z1_s0 via vanish
            z1_s = self._vanish_impl(list(s1))
            self.z1_s0 = z1_s[0::2]
        elif n == 2:
            self.z0_s1 = [fsub(s1[0], s0[0])]
            self.z1_s0 = [fsub(s0[0], s1[0])]

        self.z0_inv_s1 = batch_inv(list(self.z0_s1)) if self.z0_s1 else []
        self.z1_inv_s0 = batch_inv(list(self.z1_s0)) if self.z1_s0 else []

        # --- z0z0_rem_xnn, z1z1_rem_xnn (for EXIT's modular reduction) ---
        self.z0z0_rem_xnn_s = []
        self.z1z1_rem_xnn_s = []
        if n > 2:
            nnnn = n // 4
            xnnnn_s     = [fpow(x, nnnn) for x in s]
            xnnnn_s_inv = batch_inv(list(xnnnn_s))
            st = self.subtree
            # z0z0_rem_xnnnn_s0
            z0_rem_xnnnn_sq_s0 = [fmul(a, b)
                                  for a, b in zip(st.z0z0_rem_xnn_s, st.z1z1_rem_xnn_s)]
            z0z0_rem_xnnnn_s0 = st._modular_reduce_impl(
                z0_rem_xnnnn_sq_s0, st.xnn_s, st.z0z0_rem_xnn_s)
            z0z0_rem_xnnnn_s1 = self._extend_impl(z0z0_rem_xnnnn_s0, 'S1')
            z0z0_rem_xnnnn_s = []
            for a, b in zip(z0z0_rem_xnnnn_s0, z0z0_rem_xnnnn_s1):
                z0z0_rem_xnnnn_s.extend([a, b])
            # z0_s interleaved: [0, z0_s1[0], 0, z0_s1[1], ...]
            z0_s = []
            for y in self.z0_s1:
                z0_s.extend([0, y])
            z0_rem_xnn_s = [fsub(z, x) for z, x in zip(z0_s, self.xnn_s)]
            z0_rem_xnn_sq_s = [fmul(y, y) for y in z0_rem_xnn_s]
            z0_rem_xnn_sq_div_xnnnn_s = [
                fmul(fsub(sq, rem), inv)
                for sq, rem, inv in zip(z0_rem_xnn_sq_s, z0z0_rem_xnnnn_s, xnnnn_s_inv)
            ]
            z0z0_div_xnnnn_rem_xnnnn_s = self._modular_reduce_impl(
                z0_rem_xnn_sq_div_xnnnn_s, xnnnn_s, z0z0_rem_xnnnn_s)
            self.z0z0_rem_xnn_s = [
                fadd(rem, fmul(xn, div))
                for rem, div, xn in zip(z0z0_rem_xnnnn_s, z0z0_div_xnnnn_rem_xnnnn_s, xnnnn_s)
            ]
            # z1z1
            z1_s = []
            for y in self.z1_s0:
                z1_s.extend([y, 0])
            z1_rem_xnn_s = [fsub(z, x) for z, x in zip(z1_s, self.xnn_s)]
            z1z1 = [fmul(y, y) for y in z1_rem_xnn_s]
            self.z1z1_rem_xnn_s = self._modular_reduce_impl(
                z1z1, self.xnn_s, self.z0z0_rem_xnn_s)
        elif n == 2:
            self.z0z0_rem_xnn_s = [fmul(s0[0], s0[0])] * 2
            self.z1z1_rem_xnn_s = [fmul(s1[0], s1[0])] * 2

    # ───────────────────────────────────────────────────────────────────
    # §6.1  Subtree derivation  (mirrors derive_subtree)
    # ───────────────────────────────────────────────────────────────────

    def _derive_subtree(self):
        """
        Build the subtree by taking even-indexed elements from every layer
        and dropping the last rational map.
        """
        n = self.f.n
        if n < 2:
            return None
        sub_n = n // 2
        sub_data = [0] * (2 * sub_n)
        sub_tree = BinaryTree(sub_data)
        parent_layers = self.f.get_layers()
        # For each layer, take even-indexed elements
        log_sub = sub_n.bit_length() - 1
        for k in range(log_sub + 1):
            parent_layer = parent_layers[k]
            sub_layer = parent_layer[0::2]
            sub_tree.set_layer(k, sub_layer)
        sub_maps = self.rational_maps[:-1]

        # Build a subtree FFTree manually (avoid re-running new's assertions)
        st = object.__new__(FFTree)
        st.f = sub_tree
        st.rational_maps = sub_maps
        st._from_tree()
        return st

    # ───────────────────────────────────────────────────────────────────
    # §6.2  EXTEND  (mirrors extend_impl)
    # ───────────────────────────────────────────────────────────────────

    def _extend_impl(self, evals, moiety):
        """
        Extend evaluations from one moiety to the other.
        Mirrors ``extend_impl`` in fftree.rs.

        Given ``evals`` of length m — evaluations of a polynomial (deg < m) on
        moiety ``moiety`` — compute its evaluations on the *other* moiety.

        Algorithm (Lemma 3.2 of ECFFT I):

        1. **Decompose**: Apply M⁻¹ matrices to split into two half-vectors.
        2. **Recurse**: Extend each half on the subtree.
        3. **Recombine**: Apply M matrices to merge into the other moiety.
        """
        m = len(evals)
        if m == 1:
            return list(evals)

        # Matrix layer index: matches Rust's (num_layers - 2 - m.ilog2())
        log_n_leaves = (self.f.n).bit_length() - 1     # log2(n)
        log_m = m.bit_length() - 1                     # log2(m)
        layer = log_n_leaves - 1 - log_m

        half = m // 2

        # Decompose: select the right matrices
        # Rust: decompose_matrices.get_layer(layer), skip(1 if S0 else 0), step_by(2)
        # That selects every other matrix from the layer.
        d_layer = self.decompose[layer]
        if moiety == 'S0':
            d_mats = d_layer[1::2]    # skip 1, step 2
        else:
            d_mats = d_layer[0::2]    # skip 0, step 2

        evals0 = [0] * half
        evals1 = [0] * half
        for i in range(half):
            m_inv = d_mats[i]
            evals0[i], evals1[i] = m_inv * (evals[i], evals[i + half])

        # Recurse
        evals0p = self._extend_impl(evals0, moiety)
        evals1p = self._extend_impl(evals1, moiety)

        # Recombine: select the right matrices (opposite skip)
        r_layer = self.recombine[layer]
        if moiety == 'S0':
            r_mats = r_layer[0::2]    # skip 0, step 2
        else:
            r_mats = r_layer[1::2]    # skip 1, step 2

        res = [0] * m
        for i in range(half):
            m_r = r_mats[i]
            res[i], res[i + half] = m_r * (evals0p[i], evals1p[i])
        return res

    def extend(self, evals, moiety):
        """Extend evals (size m) from one moiety to the other. Public API."""
        tree = self._subtree_with_size(len(evals) * 2)
        return tree._extend_impl(evals, moiety)

    # ───────────────────────────────────────────────────────────────────
    # §6.3  MEXTEND  (monic polynomial extension)
    # ───────────────────────────────────────────────────────────────────

    def _mextend_impl(self, evals, moiety):
        e = self._extend_impl(evals, moiety)
        z = self.z0_s1 if moiety == 'S1' else self.z1_s0
        return [fadd(a, b) for a, b in zip(e, z)]

    # ───────────────────────────────────────────────────────────────────
    # §6.4  ENTER  (coefficients → evaluations)
    # ───────────────────────────────────────────────────────────────────

    def _enter_impl(self, coeffs):
        """
        ENTER: coefficient representation → evaluation representation.

        Mirrors ``enter_impl`` in fftree.rs.

        Given polynomial  f(x) = Σ cᵢ xⁱ  with  deg f < n:

        1. Split:  f = u(x) + x^{n/2} · v(x)
        2. Recursively evaluate u, v on the subtree domain (= ψ-images of S₀)
        3. Extend those evaluations from S₀ to S₁
        4. Recombine:  f(sⱼ) = u(sⱼ) + sⱼ^{n/2} · v(sⱼ)

        The output is in leaf order:
            [f(leaf[0]), f(leaf[1]), f(leaf[2]), …, f(leaf[n-1])]
        where even-indexed leaves are S₀ and odd-indexed are S₁.
        """
        n = len(coeffs)
        if n == 1:
            return list(coeffs)

        half = n // 2
        st = self.subtree

        # u, v on the subtree (= S₀ after ψ)
        u0 = st.enter(coeffs[:half])
        v0 = st.enter(coeffs[half:])

        # Extend from S₀ to S₁
        u1 = self._extend_impl(u0, 'S1')
        v1 = self._extend_impl(v0, 'S1')

        # Recombine — interleave S₀ and S₁ evaluations
        res = []
        for i in range(half):
            res.append(fadd(u0[i], fmul(v0[i], self.xnn_s[2 * i])))
            res.append(fadd(u1[i], fmul(v1[i], self.xnn_s[2 * i + 1])))
        return res

    def enter(self, coeffs):
        """
        ENTER: coefficients → evaluations on the domain.

        ``coeffs`` is [c₀, c₁, …, c_{n-1}].
        Returns [f(s₀), f(s₁), …, f(s_{n-1})] in leaf order.
        """
        tree = self._subtree_with_size(len(coeffs))
        return tree._enter_impl(coeffs)

    # ───────────────────────────────────────────────────────────────────
    # §6.5  REDC and MOD  (modular reduction in eval-space)
    # ───────────────────────────────────────────────────────────────────

    def _redc_impl(self, evals, a, moiety):
        """
        REDC: compute ⟨P · Z^{-1} mod a ≀ S⟩.
        Mirrors ``redc_impl`` in fftree.rs.
        """
        e0 = evals[0::2]
        e1 = evals[1::2]
        a0 = [a[i] for i in range(0, len(a), 2)]
        a1 = [a[i] for i in range(1, len(a), 2)]
        a0_inv = batch_inv(list(a0))

        # ⟨π/a ≀ S₀⟩
        t0 = [fmul(e, ai) for e, ai in zip(e0, a0_inv)]
        opp = 'S0' if moiety == 'S1' else 'S1'
        g1 = self._extend_impl(t0, opp)

        z_inv = self.z0_inv_s1 if moiety == 'S0' else self.z1_inv_s0

        # ⟨(π − a·g) / Z ≀ S'⟩
        h1 = [fmul(fsub(e, fmul(a_v, g)), zi)
              for e, g, a_v, zi in zip(e1, g1, a1, z_inv)]
        h0 = self._extend_impl(h1, moiety)

        res = []
        for a, b in zip(h0, h1):
            res.extend([a, b])
        return res

    def _modular_reduce_impl(self, evals, a, c):
        """MOD: evals mod a, using precomputed c = ⟨Z₀² mod a ≀ S⟩."""
        h = self._redc_impl(evals, a, 'S0')
        hc = [fmul(hi, ci) for hi, ci in zip(h, c)]
        return self._redc_impl(hc, a, 'S0')

    # ───────────────────────────────────────────────────────────────────
    # §6.6  EXIT  (evaluations → coefficients)
    # ───────────────────────────────────────────────────────────────────

    def _exit_impl(self, evals):
        """
        EXIT: evaluation representation → coefficient representation.

        Mirrors ``exit_impl`` in fftree.rs.

        Given evaluations [f(s₀), f(s₁), …, f(s_{n-1})] in leaf order,
        recover the unique polynomial  f  of degree < n  and return its
        coefficients [c₀, c₁, …, c_{n-1}].

        Algorithm:
        1. Modular-reduce to get  ⟨f mod x^{n/2} ≀ S₀⟩  (the low-half polynomial)
        2. Recursively interpolate the low half → coefficients a
        3. Compute  v₀[i] = (evals[2i] − u₀[i]) / s₀[i]^{n/2}
        4. Recursively interpolate the high half → coefficients b
        5. Return  a ∥ b
        """
        n = len(evals)
        if n == 1:
            return list(evals)

        # u0 = ⟨f mod X^{n/2} ≀ S₀⟩   via modular reduction
        u0 = self._modular_reduce_impl(evals, self.xnn_s, self.z0z0_rem_xnn_s)[0::2]

        st = self.subtree
        a = st._exit_impl(u0)

        # v0[i] = (evals[2i] - u0[i]) / xnn_s[2i]
        e0 = evals[0::2]
        xnn0_inv = self.xnn_s_inv[0::2]
        v0 = [fmul(fsub(e, u), xi) for e, u, xi in zip(e0, u0, xnn0_inv)]

        b = st._exit_impl(v0)

        return a + b

    def exit(self, evals):
        """
        EXIT: evaluations → coefficients.

        ``evals`` is [f(s₀), f(s₁), …] in leaf order.
        Returns [c₀, c₁, …, c_{n-1}].
        """
        tree = self._subtree_with_size(len(evals))
        return tree._exit_impl(evals)

    # ───────────────────────────────────────────────────────────────────
    # §6.7  DEGREE
    # ───────────────────────────────────────────────────────────────────

    def _degree_impl(self, evals):
        n = len(evals)
        if n == 1: return 0
        st = self.subtree
        e0 = evals[0::2]
        e1 = evals[1::2]
        g1 = self._extend_impl(e0, 'S1')
        if g1 == e1:
            return st._degree_impl(e0)
        t1 = [fmul(fsub(e, g), zi)
              for e, g, zi in zip(e1, g1, self.z0_inv_s1)]
        t0 = self._extend_impl(t1, 'S0')
        return n // 2 + st._degree_impl(t0)

    def degree(self, evals):
        tree = self._subtree_with_size(len(evals))
        return tree._degree_impl(evals)

    # ───────────────────────────────────────────────────────────────────
    # §6.8  VANISH
    # ───────────────────────────────────────────────────────────────────

    def _vanish_impl(self, vanish_domain):
        n = len(vanish_domain)
        if n == 1:
            l = self.f.leaves()
            assert len(l) == 2
            alpha = vanish_domain[0]
            return [fsub(alpha, l[0]), fsub(alpha, l[1])]
        st = self.subtree
        qp  = st._vanish_impl(vanish_domain[:n // 2])
        qpp = st._vanish_impl(vanish_domain[n // 2:])
        q_s0 = [fmul(a, b) for a, b in zip(qp, qpp)]
        q_s1 = self._mextend_impl(q_s0, 'S1')
        res = []
        for a, b in zip(q_s0, q_s1):
            res.extend([a, b])
        return res

    # ───────────────────────────────────────────────────────────────────
    # §6.9  Navigation
    # ───────────────────────────────────────────────────────────────────

    def _subtree_with_size(self, n):
        """Return the (sub)tree whose leaf count equals n."""
        if n == self.f.n:
            return self
        if n < self.f.n:
            assert self.subtree is not None
            return self.subtree._subtree_with_size(n)
        raise ValueError("FFTree too small")

    def eval_domain(self):
        """Return the evaluation domain (leaf x-coordinates) in order."""
        return list(self.f.leaves())


# ═══════════════════════════════════════════════════════════════════════════
# §7  Building an FFTree from curve parameters
# ═══════════════════════════════════════════════════════════════════════════

def build_fftree(params, log_n):
    """
    Build an FFTree of size 2^{log_n} from curve parameters.

    ``params`` must have keys: 'a', 'bb', 'gx', 'gy', 'k'.

    Returns (fftree, domain) where domain = list of evaluation x-coords.
    """
    assert log_n <= params['k']
    n = 1 << log_n

    curve = GoodCurve(params['a'], params['bb'])
    gen   = Point(params['gx'], params['gy'], curve)

    # Scale generator to order n
    scaled_gen = gen.scalar_mul(1 << (params['k'] - log_n))

    # Build isogeny chain → rational maps
    psis, curves, hs = build_isogeny_chain(scaled_gen, log_n)

    # Build evaluation domain:  x(O + i·G) for i = 0, …, n-1
    # Use 2*gen as coset offset
    coset = gen.double()
    leaves = []
    acc = Point.infinity()
    for _ in range(n):
        leaves.append((coset + acc).x)
        acc = acc + scaled_gen

    tree = FFTree(leaves, psis)
    return tree, leaves


# ═══════════════════════════════════════════════════════════════════════════
# §8  Convenience:  split_domain_with_psi  (for demos)
# ═══════════════════════════════════════════════════════════════════════════

def build_evaluation_domain(generator, coset_offset, n):
    assert n > 0 and (n & (n - 1)) == 0
    domain, acc = [], Point.infinity()
    for _ in range(n):
        domain.append((coset_offset + acc).x)
        acc = acc + generator
    return domain


def split_domain_with_psi(domain, psi):
    """Split domain into S₀ (first half) and S₁ (second half) via ψ."""
    half = len(domain) // 2
    s0, s1 = domain[:half], domain[half:]
    images = []
    for i in range(half):
        img0, img1 = psi(s0[i]), psi(s1[i])
        assert img0 == img1, f"ψ pairing broken at {i}: {img0} ≠ {img1}"
        images.append(img0)
    return s0, s1, images


# ═══════════════════════════════════════════════════════════════════════════
# §9  Demo / verification
# ═══════════════════════════════════════════════════════════════════════════

def demo(params, size=8):
    """
    Full ECFFT demo: build tree, ENTER, EXIT, round-trip.
    """
    log_n = size.bit_length() - 1
    assert 1 << log_n == size
    print(f"{'='*72}")
    print(f"ECFFT Demo — 2^{params['k']} curve, domain size {size}")
    print(f"{'='*72}")

    tree, leaves = build_fftree(params, log_n)
    domain = tree.eval_domain()
    print(f"  Domain: {domain[:4]}{'…' if size > 4 else ''}")

    # --- ENTER ---
    coeffs = [(i + 1) % q for i in range(size)]
    evals = tree.enter(coeffs)
    naive  = [poly_eval(coeffs, x) for x in domain]
    enter_ok = evals == naive
    print(f"  ENTER {'✓' if enter_ok else '✗'}  (matches naive eval: {enter_ok})")
    if not enter_ok:
        for i in range(size):
            if evals[i] != naive[i]:
                print(f"    [{i}] ecfft={evals[i]}  naive={naive[i]}")

    # --- EXIT ---
    recovered = tree.exit(evals)
    exit_ok = recovered == coeffs
    print(f"  EXIT  {'✓' if exit_ok else '✗'}  (round-trip: {exit_ok})")
    if not exit_ok:
        for i in range(size):
            if recovered[i] != coeffs[i]:
                print(f"    [{i}] got={recovered[i]}  want={coeffs[i]}")

    # --- DEGREE ---
    deg = tree.degree(evals)
    print(f"  DEGREE ✓  deg={deg}  (expected {size - 1})" if deg == size - 1
          else f"  DEGREE ✗  deg={deg}  (expected {size - 1})")

    print(f"{'='*72}")
    return enter_ok and exit_ok


# ═══════════════════════════════════════════════════════════════════════════
# §10  Naive helpers (for cross-checking)
# ═══════════════════════════════════════════════════════════════════════════

def lagrange_interpolate(xs, ys):
    """O(n²) Lagrange interpolation → coefficient list."""
    n = len(xs)
    coeffs = [0] * n
    for j in range(n):
        basis = [1]
        denom = 1
        for k in range(n):
            if k == j: continue
            denom = fmul(denom, fsub(xs[j], xs[k]))
            new = [0] * (len(basis) + 1)
            for m in range(len(basis)):
                new[m]     = fadd(new[m], fmul(basis[m], fneg(xs[k])))
                new[m + 1] = fadd(new[m + 1], basis[m])
            basis = new
        scale = fdiv(ys[j], denom)
        for m in range(len(basis)):
            coeffs[m] = fadd(coeffs[m], fmul(basis[m], scale))
    return coeffs


def verify_two_to_one(psi, domain_xs):
    images = {}
    for x in domain_xs:
        img = psi(x)
        images.setdefault(img, []).append(x)
    for img, pre in sorted(images.items()):
        assert len(pre) == 2, f"Expected 2 preimages for {img}, got {len(pre)}"
    return [img for img in sorted(images.keys())]


# ═══════════════════════════════════════════════════════════════════════════
# §11  Global ECFFT decomposition (Part I style)
# ═══════════════════════════════════════════════════════════════════════════
#
# This section provides the ECFFT Part I global decomposition:
#
#     f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x))
#
# where  u  has the low-half coefficients (degree < n/2),  v  the high-half,
# and  ψ  is the 2-to-1 rational map from the good isogeny.
#
# IMPORTANT: This decomposition is GLOBAL — it requires the full FFTree
# modular reduction machinery (MOD = REDC∘REDC) and is O(n log n).
# It is NOT the same as the ECFFT Part II FRI hash (see §12 below).
#
# The result lives on the SUBTREE domain (even-indexed leaves), which is
# NOT the same as the ψ-image domain. This is fine for ECFFT ENTER/EXIT
# but is NOT suitable as a verifier-checkable FRI round relation.
#
# For FRI-style protocols, use the functions in §12 instead.
#
# Functions:
#   ecfft_decompose_step(evals, tree) → (u_evals, v_evals)
#   ecfft_fold_step(evals, tree, α)  → folded_evals
#   ecfft_fold(evals, tree, [α₁, α₂, …]) → fully-folded evals
# ═══════════════════════════════════════════════════════════════════════════


def ecfft_decompose_step(evals, tree):
    """
    Global ECFFT Part I decomposition: f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x)).

    This uses the FFTree's modular reduction machinery (O(n log n), global).
    Result lives on the SUBTREE domain (even-indexed leaves), NOT on ψ-images.

    For FRI protocols, prefer ecfri_fold_step() from §12, which is pointwise
    and produces output on the correct ψ-image domain.

    Parameters
    ----------
    evals : list of int
        [f(s₀), f(s₁), …, f(s_{n-1})] in leaf order (size n).
    tree : FFTree
        The FFTree whose domain matches the evaluations.

    Returns
    -------
    u_evals, v_evals : each list of int, size n/2
        Evaluations on the subtree domain.
    """
    n = len(evals)
    t = tree._subtree_with_size(n)

    # Step 1: ⟨f mod x^{n/2} ≀ S₀⟩  via modular reduction
    u_evals = t._modular_reduce_impl(evals, t.xnn_s, t.z0z0_rem_xnn_s)[0::2]

    # Step 2: v₀[i] = (f(S₀[i]) − u(S₀[i])) / S₀[i]^{n/2}
    e0 = evals[0::2]
    xnn0_inv = t.xnn_s_inv[0::2]
    v_evals = [fmul(fsub(e, u), xi) for e, u, xi in zip(e0, u_evals, xnn0_inv)]

    return u_evals, v_evals


def ecfft_fold_step(evals, tree, alpha):
    """
    Global fold: f_folded = u + α·v on the subtree domain.

    WARNING: This is the Part I global decomposition, NOT the ECFFT2 FRI hash.
    The fold is O(n log n) and the result lives on the subtree domain (even-
    indexed leaves), which differs from the ψ-image domain. This means:
      - The fold matrix is DENSE (every output depends on all inputs).
      - A verifier CANNOT check a single query in O(1).

    For FRI protocols, use ecfri_fold_step() from §12 instead.
    """
    u_evals, v_evals = ecfft_decompose_step(evals, tree)
    return [fadd(u, fmul(alpha, v)) for u, v in zip(u_evals, v_evals)]


def ecfft_fold(evals, tree, alphas):
    """
    Multi-round global fold. See ecfft_fold_step() for caveats.

    For FRI protocols, use ecfri_fold() from §12 instead.
    """
    current = list(evals)
    t = tree._subtree_with_size(len(current))
    for alpha in alphas:
        current = ecfft_fold_step(current, t, alpha)
        if t.subtree is not None:
            t = t.subtree
    return current


# ═══════════════════════════════════════════════════════════════════════════
# §12  ECFFT Part II FRI hash — the correct pointwise fold
# ═══════════════════════════════════════════════════════════════════════════
#
# This section implements the FRI algebraic hash H_z from ECFFT Part II
# (BSCKL22, Appendix B.2). Unlike the global decomposition in §11, this
# hash is:
#
#   1. POINTWISE: each output depends on exactly 2 inputs (a ψ-preimage pair).
#   2. DEGREE-AWARE: it depends on the current degree bound d, not just |L_i|.
#   3. Lives on the ψ-IMAGE DOMAIN: L_{i+1} = ψ_i(L_i), not the subtree.
#
# ── Mathematical basis ──
#
# The paper (ECFFT Part I, Lemma 3.2) shows that for a degree-2 rational
# map ψ(x) = u(x)/v(x), any polynomial P of degree < d decomposes as:
#
#   P(x) = (P_0(ψ(x)) + x · P_1(ψ(x))) · v(x)^{d/2 - 1}
#
# with deg(P_0), deg(P_1) < d/2. For our good isogeny ψ(x) = (x-b)²/x,
# the denominator is v(x) = x, so the normalization factor is x^{d/2 - 1}.
#
# For a pair (s_0, s_1) with ψ(s_0) = ψ(s_1) = t, setting e = d/2 - 1:
#
#   P(s_0) / s_0^e = P_0(t) + s_0 · P_1(t)
#   P(s_1) / s_1^e = P_0(t) + s_1 · P_1(t)
#
# This is a 2×2 system. The FRI hash evaluates P_0 + z·P_1 at t:
#
#   H_z[P](t) = a + slope · (z - s_0)
#
# where:
#   a     = P(s_0) / s_0^e
#   b     = P(s_1) / s_1^e
#   slope = (b - a) / (s_1 - s_0)
#
# ── Domain structure ──
#
# The domains L_0, L_1, ..., L_k are connected by ψ:
#   L_{i+1} = {ψ_i(x) : x ∈ L_i}
#
# At each layer, pairing is FIRST-HALF / SECOND-HALF:
#   ψ(L_i[j]) = ψ(L_i[j + m/2]) = L_{i+1}[j]    for j < m/2
#
# where m = |L_i|. This is verified during tree construction (the tree
# stores L_i as get_layer(i), and L_{i+1}[j] = rational_maps[i](L_i[j])).
#
# ── Key difference from §11 ──
#
# The §11 global decomposition uses basis {1, x, x², ..., x^{n/2-1}}
# and produces output on the subtree domain (even-indexed leaves).
#
# The §12 FRI hash uses basis {v(x)^e, x·v(x)^e} (degree-aware) and
# produces output on the ψ-image domain. The two give DIFFERENT polynomials
# for the same input, but both are valid degree-halving operations.
#
# Only §12 is suitable for FRI verification (pointwise checking).
# ═══════════════════════════════════════════════════════════════════════════


def build_fri_domains(params, log_n):
    """
    Build the FRI layer domains L_0, ..., L_k from curve parameters.

    Returns a list of layers: layers[i] is a list of field elements (size 2^{log_n - i}).

    The pairing invariant holds: ψ_i(L_i[j]) = ψ_i(L_i[j + m/2]) = L_{i+1}[j].

    Also returns the list of rational maps (isogenies).

    Parameters
    ----------
    params : dict
        Curve parameters with keys 'a', 'bb', 'gx', 'gy', 'k'.
    log_n : int
        Log2 of the initial domain size.

    Returns
    -------
    layers : list of list of int
        layers[i] has size 2^{log_n - i}.
    rational_maps : list of RationalMap
        rational_maps[i] maps L_i → L_{i+1}.
    """
    assert log_n <= params['k']
    n = 1 << log_n

    curve = GoodCurve(params['a'], params['bb'])
    gen = Point(params['gx'], params['gy'], curve)
    scaled_gen = gen.scalar_mul(1 << (params['k'] - log_n))

    # Build isogeny chain
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


def ecfri_fold_step(word, layers, round_idx, degree_bound, z):
    """
    ECFFT Part II FRI hash: the correct pointwise fold for FRI protocols.

    For each pair (s_0, s_1) = (L_i[j], L_i[j + m/2]) with degree bound d:

        e = d/2 - 1
        a = word[j] / s_0^e              (normalize by v(s_0)^e, v(x) = x)
        b = word[j + m/2] / s_1^e
        slope = (b - a) / (s_1 - s_0)
        out[j] = a + slope · (z - s_0)   (= P_0(t) + z · P_1(t))

    This is POINTWISE: out[j] depends only on word[j] and word[j + m/2].

    Parameters
    ----------
    word : list of int
        Evaluations on layer L_i (size m = |L_i|).
    layers : list of list of int
        The FRI domain layers from build_fri_domains().
    round_idx : int
        Current round index (0-based). Uses layers[round_idx] as the domain.
    degree_bound : int
        Current degree bound d_i. Must be even, ≤ m.
    z : int
        Verifier challenge.

    Returns
    -------
    out : list of int
        Evaluations on layer L_{i+1} (size m/2).
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
    Multi-round ECFFT Part II FRI fold.

    Parameters
    ----------
    word : list of int
        Evaluations on layers[0].
    layers : list of list of int
        The FRI domain layers from build_fri_domains().
    degree_bound : int
        Initial degree bound (halved each round).
    challenges : list of int
        One challenge per round.

    Returns
    -------
    folded : list of int
        Evaluations on layers[len(challenges)] (size n / 2^len(challenges)).
    """
    current = list(word)
    d = degree_bound
    for i, z in enumerate(challenges):
        current = ecfri_fold_step(current, layers, i, d, z)
        d = d // 2
    return current


def ecfri_verify_query(layers, round_idx, degree_bound, j, f_s0, f_s1, z):
    """
    Verifier: compute expected fold value from a single pair opening.

    Given f(s_0) and f(s_1) for pair index j in round round_idx, returns
    the expected value at layers[round_idx + 1][j].

    This is the O(1) per-query check that makes FRI efficient.

    Parameters
    ----------
    layers : list of list of int
    round_idx : int
    degree_bound : int
    j : int
        Pair index (0 ≤ j < |L_i|/2).
    f_s0, f_s1 : int
        The two opened evaluations: f(L_i[j]) and f(L_i[j + m/2]).
    z : int
        Verifier challenge for this round.

    Returns
    -------
    expected : int
        The expected fold value at layers[round_idx + 1][j].
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
# §13  Group-valued BaseFold (eprint 2025/1325, Section 7)
# ═══════════════════════════════════════════════════════════════════════════
#
# This section implements the group-valued BaseFold protocol from
# "Revisiting the IPA-sumcheck connection" (Eagen & Gabizon, 2025).
#
# ── What it replaces ──
#
# In IPA verification, the "decide" step computes:
#   G(r) = Σ_{i<n} s_i · G_i
# where G_0,...,G_{n-1} are SRS generators and s_i are derived from IPA
# challenges. This is an O(n) MSM that dominates recursive verification.
#
# BaseFold replaces this with a FRI-like protocol over GROUP ELEMENTS:
# - The prover commits Merkle trees of group-element evaluations
# - The verifier spot-checks fold consistency at random positions
# - Each spot-check costs O(1) scalar multiplications
#
# ── Protocol overview ──
#
# 1. SRS ENCODING (one-time precomputation):
#    g_0[j] = Σ_{i<n} L_0[j]^i · G_i   for j = 0,...,|L_0|-1
#    This evaluates the "group polynomial" G(x) = Σ s_i·x^i·G_i on L_0.
#    (For BaseFold, the s_i are IPA challenge-derived scalars.)
#
# 2. FRI ROUNDS (k = log_n rounds):
#    For round i = 0,...,k-1:
#      a. Prover commits g_i (Merkle root of group elements on L_i)
#      b. Verifier sends challenge z_i
#      c. Prover computes g_{i+1} by applying the ECFFT2 fold to g_i
#         (pointwise: each pair produces one output via 4 scalar muls)
#
# 3. FINAL CHECK:
#    g_k is a single group element. Verifier checks it matches the
#    expected final folded value.
#
# 4. QUERY PHASE:
#    Verifier picks ~43 random query indices. For each query, traces
#    through all k rounds, opening the Merkle commitment at each pair
#    and checking fold consistency:
#
#      e = d_i/2 - 1
#      a = g_i[j] * (1/s_0^e)           (scalar mul)
#      b = g_i[j + m/2] * (1/s_1^e)     (scalar mul)
#      slope = (b - a) * diff_inv        (scalar mul)
#      expected = a + slope * (z - s_0)  (scalar mul)
#      CHECK: expected == g_{i+1}[j']    (group element equality)
#
#    Total: 4 scalar muls + Merkle path verification per round per query.
#
# ── Group element representation ──
#
# Group elements are (x, y) pairs on the Grumpkin curve (= BN254 G1 with
# base field = BN254 Fq). Scalar multiplication and addition are the
# standard elliptic curve operations.
#
# For Merkle commitments, each leaf is hash(x, y) using Poseidon2.
# ═══════════════════════════════════════════════════════════════════════════


def basefold_group_fold_step(g_word, layers, round_idx, degree_bound, z):
    """
    BaseFold prover: fold a group-element oracle using the ECFFT2 hash.

    Same formula as ecfri_fold_step, but operating on (x, y) curve points
    instead of field elements. "Division" by a scalar becomes multiplication
    by the scalar inverse; "addition" is group addition.

    Parameters
    ----------
    g_word : list of (int, int)
        Group elements (x, y) on layer L_i (size m).
        Each element is a pair of field elements (affine coordinates).
    layers : list of list of int
        The FRI domain layers from build_fri_domains().
    round_idx : int
        Current round.
    degree_bound : int
        Current degree bound.
    z : int
        Verifier challenge.

    Returns
    -------
    g_out : list of (int, int)
        Group elements on layer L_{i+1} (size m/2).
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

        # a = g_word[j] * (1/s_0^e)  — scalar mul on group element
        a = _group_scalar_mul(g_word[j], s0_e_inv)
        # b = g_word[j + half] * (1/s_1^e)
        b = _group_scalar_mul(g_word[j + half], s1_e_inv)

        # slope = (b - a) * diff_inv  — group subtraction then scalar mul
        b_minus_a = _group_add(b, _group_neg(a))
        slope = _group_scalar_mul(b_minus_a, diff_invs[j])

        # out = a + slope * (z - s_0)
        z_minus_s0 = fsub(z, layer[j])
        g_out[j] = _group_add(a, _group_scalar_mul(slope, z_minus_s0))

    return g_out


def basefold_verify_query(layers, round_idx, degree_bound, j, g_s0, g_s1, z):
    """
    BaseFold verifier: check a single fold query over group elements.

    Same as ecfri_verify_query but over group elements. Returns the
    expected fold value as a group element.

    Parameters
    ----------
    layers : list of list of int
    round_idx, degree_bound, j, z : as in ecfri_verify_query
    g_s0, g_s1 : (int, int)
        Opened group elements at the pair.

    Returns
    -------
    expected : (int, int)
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


# ── Group element helpers (simple affine arithmetic over Fq) ──
# These operate on tuples (x, y) or None for the point at infinity.

def _group_add(p, q_pt):
    """Add two affine points on a Weierstrass curve (or None for infinity)."""
    if p is None:
        return q_pt
    if q_pt is None:
        return p
    px, py = p
    qx, qy = q_pt
    if px == qx:
        if py == qy and py != 0:
            # Point doubling — we need the curve 'a' parameter.
            # For Grumpkin: y² = x³ + b (a=0), so λ = 3x²/(2y).
            # For general Montgomery: caller should provide curve params.
            # Using a=0 Weierstrass for now (Grumpkin).
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
    print("This module provides the FFTree class and ECFFT algorithms.")
    print("Import a parameter file and call demo(params) to test.")
    print()
    print("  from ecfft_params_2_18 import params")
    print("  from ecfft_algorithms import demo")
    print("  demo(params, size=8)")
