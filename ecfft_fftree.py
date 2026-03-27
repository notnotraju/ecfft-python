"""
ECFFT General-Purpose FFTree
==============================

This file contains the general-purpose ECFFT machinery: the FFTree data
structure with its ENTER, EXIT, EXTEND, and DEGREE operations, plus the
global Part I decomposition ("Milson-style folding").

These are **not** needed for the group BaseFold protocol — they implement
the full polynomial evaluation/interpolation engine from ECFFT Part I.
For group BaseFold, see ``ecfft_algorithms.py``.

The FFTree provides O(n log² n) polynomial evaluation (ENTER) and
interpolation (EXIT) over any prime field, without requiring roots of unity.

References:

  * **ECFFT Part I**:  https://arxiv.org/pdf/2107.08473.pdf
  * **Rust implementation**: https://github.com/andrewmilson/ecfft


Usage
-----

::

    from ecfft_fftree import build_fftree, FFTree
    from ecfft_params_2_20 import params

    tree, leaves = build_fftree(params, log_n=5)

    coeffs = list(range(1, 33))
    evals = tree.enter(coeffs)          # O(n log² n) evaluation
    recovered = tree.exit(evals)        # O(n log² n) interpolation
    assert recovered == coeffs

    deg = tree.degree(evals)            # O(n log n) degree computation
    assert deg == 31


Global ECFFT decomposition (Part I style)
------------------------------------------

Also provided here is the Part I global decomposition:

    f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x))

This uses the FFTree's modular reduction machinery — it is O(n log n) and
**not pointwise**.  Each output depends on all inputs, so it cannot be
verified per-query in O(1).  This is fine for ENTER/EXIT internals but is
**not suitable for FRI verification**.

For FRI protocols, use ``ecfri_fold_step`` from ``ecfft_algorithms.py``.
"""

from ecfft_algorithms import (
    q, fadd, fsub, fmul, fdiv, fneg, finv, fpow, fsqrt,
    batch_inv, poly_eval,
    GoodCurve, Point, RationalMap,
    good_isogeny, apply_isogeny, build_isogeny_chain,
)


# ═══════════════════════════════════════════════════════════════════════════
# BinaryTree — heap-layout binary tree
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
    def num_layers(self):    return self.n.bit_length()

    def get_layer(self, i):
        """Layer i: layer 0 = leaves (size n), layer 1 = size n/2, …"""
        sz = self.n >> i
        return self.data[sz : 2 * sz]

    def set_layer(self, i, vals):
        sz = self.n >> i
        self.data[sz : 2 * sz] = vals

    def get_layers(self):
        """Return [layer_0 (leaves), layer_1, …, layer_{d-1} (root)]."""
        log_n = (self.n - 1).bit_length() if self.n > 1 else 0
        return [self.get_layer(i) for i in range(log_n + 1)]


# ═══════════════════════════════════════════════════════════════════════════
# Mat2x2 — 2×2 matrices over Fq
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
# FFTree — the core data structure
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
    domain.

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
        for k, rmap in enumerate(rational_maps):
            prev = self.f.get_layer(k)
            layer_sz = len(prev) // 2
            layer = [0] * layer_sz
            for i in range(layer_sz):
                layer[i] = rmap(prev[i])
                assert layer[i] == rmap(prev[i + layer_sz]), \
                    f"ψ pairing broken at level {k}, index {i}"
            self.f.set_layer(k + 1, layer)

        self.rational_maps = list(rational_maps)

        # --- Delegate to _from_tree for all the precomputation ---
        self._from_tree()

    def _from_tree(self):
        """Precompute everything needed for ENTER / EXIT / EXTEND / etc."""
        n = self.f.n
        log_n = n.bit_length() - 1
        s = self.f.leaves()

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
        layers = self.f.get_layers()
        self.recombine = []
        self.decompose = []

        for k in range(log_n):
            lyr = layers[k]
            d = len(lyr) // 2
            if d <= 1:
                self.recombine.append([])
                self.decompose.append([])
                continue
            rmap = self.rational_maps[k]
            v_poly = rmap.den
            exp = d // 2 - 1
            if exp < 0: exp = 0
            rmats = []
            dmats = []
            for i in range(d):
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
            st_z0_s0 = []
            st_z1_s0 = []
            for y in st.z0_s1:
                st_z0_s0.extend([0, y])
            for y in st.z1_s0:
                st_z1_s0.extend([y, 0])
            st_z0_s1 = self._extend_impl(st_z0_s0, 'S1')
            st_z1_s1 = self._extend_impl(st_z1_s0, 'S1')
            self.z0_s1 = [fmul(a, b) for a, b in zip(st_z0_s1, st_z1_s1)]
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
            z0_rem_xnnnn_sq_s0 = [fmul(a, b)
                                  for a, b in zip(st.z0z0_rem_xnn_s, st.z1z1_rem_xnn_s)]
            z0z0_rem_xnnnn_s0 = st._modular_reduce_impl(
                z0_rem_xnnnn_sq_s0, st.xnn_s, st.z0z0_rem_xnn_s)
            z0z0_rem_xnnnn_s1 = self._extend_impl(z0z0_rem_xnnnn_s0, 'S1')
            z0z0_rem_xnnnn_s = []
            for a, b in zip(z0z0_rem_xnnnn_s0, z0z0_rem_xnnnn_s1):
                z0z0_rem_xnnnn_s.extend([a, b])
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
    # Subtree derivation
    # ───────────────────────────────────────────────────────────────────

    def _derive_subtree(self):
        """Build the subtree by taking even-indexed elements from every layer."""
        n = self.f.n
        if n < 2:
            return None
        sub_n = n // 2
        sub_data = [0] * (2 * sub_n)
        sub_tree = BinaryTree(sub_data)
        parent_layers = self.f.get_layers()
        log_sub = sub_n.bit_length() - 1
        for k in range(log_sub + 1):
            parent_layer = parent_layers[k]
            sub_layer = parent_layer[0::2]
            sub_tree.set_layer(k, sub_layer)
        sub_maps = self.rational_maps[:-1]

        st = object.__new__(FFTree)
        st.f = sub_tree
        st.rational_maps = sub_maps
        st._from_tree()
        return st

    # ───────────────────────────────────────────────────────────────────
    # EXTEND
    # ───────────────────────────────────────────────────────────────────

    def _extend_impl(self, evals, moiety):
        """
        Extend evaluations from one moiety to the other.

        Given ``evals`` of length m on moiety ``moiety``, compute its
        evaluations on the other moiety via the Lemma 3.2 decompose/recurse/
        recombine strategy.
        """
        m = len(evals)
        if m == 1:
            return list(evals)

        log_n_leaves = (self.f.n).bit_length() - 1
        log_m = m.bit_length() - 1
        layer = log_n_leaves - 1 - log_m

        half = m // 2

        d_layer = self.decompose[layer]
        if moiety == 'S0':
            d_mats = d_layer[1::2]
        else:
            d_mats = d_layer[0::2]

        evals0 = [0] * half
        evals1 = [0] * half
        for i in range(half):
            m_inv = d_mats[i]
            evals0[i], evals1[i] = m_inv * (evals[i], evals[i + half])

        evals0p = self._extend_impl(evals0, moiety)
        evals1p = self._extend_impl(evals1, moiety)

        r_layer = self.recombine[layer]
        if moiety == 'S0':
            r_mats = r_layer[0::2]
        else:
            r_mats = r_layer[1::2]

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
    # MEXTEND (monic polynomial extension)
    # ───────────────────────────────────────────────────────────────────

    def _mextend_impl(self, evals, moiety):
        e = self._extend_impl(evals, moiety)
        z = self.z0_s1 if moiety == 'S1' else self.z1_s0
        return [fadd(a, b) for a, b in zip(e, z)]

    # ───────────────────────────────────────────────────────────────────
    # ENTER (coefficients → evaluations)
    # ───────────────────────────────────────────────────────────────────

    def _enter_impl(self, coeffs):
        """
        ENTER: coefficient representation → evaluation representation.

        Split f = u(x) + x^{n/2}·v(x), recursively evaluate u, v on the
        subtree domain, extend to the other moiety, recombine.
        """
        n = len(coeffs)
        if n == 1:
            return list(coeffs)

        half = n // 2
        st = self.subtree

        u0 = st.enter(coeffs[:half])
        v0 = st.enter(coeffs[half:])

        u1 = self._extend_impl(u0, 'S1')
        v1 = self._extend_impl(v0, 'S1')

        res = []
        for i in range(half):
            res.append(fadd(u0[i], fmul(v0[i], self.xnn_s[2 * i])))
            res.append(fadd(u1[i], fmul(v1[i], self.xnn_s[2 * i + 1])))
        return res

    def enter(self, coeffs):
        """ENTER: coefficients → evaluations on the domain."""
        tree = self._subtree_with_size(len(coeffs))
        return tree._enter_impl(coeffs)

    # ───────────────────────────────────────────────────────────────────
    # REDC and MOD (modular reduction in eval-space)
    # ───────────────────────────────────────────────────────────────────

    def _redc_impl(self, evals, a, moiety):
        e0 = evals[0::2]
        e1 = evals[1::2]
        a0 = [a[i] for i in range(0, len(a), 2)]
        a1 = [a[i] for i in range(1, len(a), 2)]
        a0_inv = batch_inv(list(a0))

        t0 = [fmul(e, ai) for e, ai in zip(e0, a0_inv)]
        opp = 'S0' if moiety == 'S1' else 'S1'
        g1 = self._extend_impl(t0, opp)

        z_inv = self.z0_inv_s1 if moiety == 'S0' else self.z1_inv_s0

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
    # EXIT (evaluations → coefficients)
    # ───────────────────────────────────────────────────────────────────

    def _exit_impl(self, evals):
        """
        EXIT: evaluation representation → coefficient representation.

        Modular-reduce to get u = f mod x^{n/2}, recurse, then recover v.
        """
        n = len(evals)
        if n == 1:
            return list(evals)

        u0 = self._modular_reduce_impl(evals, self.xnn_s, self.z0z0_rem_xnn_s)[0::2]

        st = self.subtree
        a = st._exit_impl(u0)

        e0 = evals[0::2]
        xnn0_inv = self.xnn_s_inv[0::2]
        v0 = [fmul(fsub(e, u), xi) for e, u, xi in zip(e0, u0, xnn0_inv)]

        b = st._exit_impl(v0)

        return a + b

    def exit(self, evals):
        """EXIT: evaluations → coefficients."""
        tree = self._subtree_with_size(len(evals))
        return tree._exit_impl(evals)

    # ───────────────────────────────────────────────────────────────────
    # DEGREE
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
    # VANISH
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
    # Navigation
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
# Building an FFTree from curve parameters
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

    scaled_gen = gen.scalar_mul(1 << (params['k'] - log_n))

    psis, curves, hs = build_isogeny_chain(scaled_gen, log_n)

    coset = gen.double()
    leaves = []
    acc = Point.infinity()
    for _ in range(n):
        leaves.append((coset + acc).x)
        acc = acc + scaled_gen

    tree = FFTree(leaves, psis)
    return tree, leaves


# ═══════════════════════════════════════════════════════════════════════════
# Convenience helpers (for demos)
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
# Naive helpers (for cross-checking)
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
# Global ECFFT decomposition (Part I style)
# ═══════════════════════════════════════════════════════════════════════════
#
# The Part I global decomposition:  f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x))
#
# IMPORTANT: This decomposition is GLOBAL — it requires the full FFTree
# modular reduction machinery (MOD = REDC∘REDC) and is O(n log n).
# The result lives on the SUBTREE domain (even-indexed leaves), NOT the
# ψ-image domain.
#
# This is NOT suitable for FRI verification (not pointwise).
# For FRI protocols, use ecfri_fold_step from ecfft_algorithms.py.
# ═══════════════════════════════════════════════════════════════════════════


def ecfft_decompose_step(evals, tree):
    """
    Global ECFFT Part I decomposition: f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x)).

    Uses the FFTree's modular reduction machinery (O(n log n), global).
    Result lives on the SUBTREE domain, NOT on ψ-images.

    For FRI protocols, prefer ecfri_fold_step() from ecfft_algorithms.py.
    """
    n = len(evals)
    t = tree._subtree_with_size(n)

    u_evals = t._modular_reduce_impl(evals, t.xnn_s, t.z0z0_rem_xnn_s)[0::2]

    e0 = evals[0::2]
    xnn0_inv = t.xnn_s_inv[0::2]
    v_evals = [fmul(fsub(e, u), xi) for e, u, xi in zip(e0, u_evals, xnn0_inv)]

    return u_evals, v_evals


def ecfft_fold_step(evals, tree, alpha):
    """
    Global fold: f_folded = u + α·v on the subtree domain.

    WARNING: This is the Part I global decomposition, NOT the ECFFT2 FRI hash.
    The fold is O(n log n), dense, and NOT pointwise-verifiable.

    For FRI protocols, use ecfri_fold_step() from ecfft_algorithms.py.
    """
    u_evals, v_evals = ecfft_decompose_step(evals, tree)
    return [fadd(u, fmul(alpha, v)) for u, v in zip(u_evals, v_evals)]


def ecfft_fold(evals, tree, alphas):
    """Multi-round global fold. See ecfft_fold_step() for caveats."""
    current = list(evals)
    t = tree._subtree_with_size(len(current))
    for alpha in alphas:
        current = ecfft_fold_step(current, t, alpha)
        if t.subtree is not None:
            t = t.subtree
    return current


# ═══════════════════════════════════════════════════════════════════════════
# Demo / verification
# ═══════════════════════════════════════════════════════════════════════════

def demo(params, size=8):
    """Full ECFFT demo: build tree, ENTER, EXIT, round-trip."""
    log_n = size.bit_length() - 1
    assert 1 << log_n == size
    print(f"{'='*72}")
    print(f"ECFFT Demo — 2^{params['k']} curve, domain size {size}")
    print(f"{'='*72}")

    tree, leaves = build_fftree(params, log_n)
    domain = tree.eval_domain()
    print(f"  Domain: {domain[:4]}{'…' if size > 4 else ''}")

    coeffs = [(i + 1) % q for i in range(size)]
    evals = tree.enter(coeffs)
    naive  = [poly_eval(coeffs, x) for x in domain]
    enter_ok = evals == naive
    print(f"  ENTER {'✓' if enter_ok else '✗'}  (matches naive eval: {enter_ok})")
    if not enter_ok:
        for i in range(size):
            if evals[i] != naive[i]:
                print(f"    [{i}] ecfft={evals[i]}  naive={naive[i]}")

    recovered = tree.exit(evals)
    exit_ok = recovered == coeffs
    print(f"  EXIT  {'✓' if exit_ok else '✗'}  (round-trip: {exit_ok})")
    if not exit_ok:
        for i in range(size):
            if recovered[i] != coeffs[i]:
                print(f"    [{i}] got={recovered[i]}  want={coeffs[i]}")

    deg = tree.degree(evals)
    print(f"  DEGREE ✓  deg={deg}  (expected {size - 1})" if deg == size - 1
          else f"  DEGREE ✗  deg={deg}  (expected {size - 1})")

    print(f"{'='*72}")
    return enter_ok and exit_ok


if __name__ == "__main__":
    print("ECFFT General-Purpose FFTree")
    print()
    print("  from ecfft_fftree import build_fftree, demo")
    print("  from ecfft_params_2_20 import params")
    print("  demo(params, size=8)")
