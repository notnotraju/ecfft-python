# ecfft-python

Pure-Python implementation of the **Elliptic Curve Fast Fourier Transform** (ECFFT) over the BN-254 base field.

This is a literate, dependency-free implementation of the ECFFT algorithms from:

- [ECFFT Part I: Fast Polynomial Algorithms over all Finite Fields](https://arxiv.org/pdf/2107.08473.pdf) (Eli Ben-Sasson, Dan Carmon, Yair Frankel, Swastik Kopparty)
- [ECFFT Part II](https://www.math.toronto.edu/swastik/ECFFT2.pdf) (Eli Ben-Sasson, Dan Carmon, Swastik Kopparty, David Levit)

Based on the Rust implementation at [andrewmilson/ecfft](https://github.com/andrewmilson/ecfft).

## Why?

The standard FFT requires the field to contain roots of unity of large 2-power order (i.e., the field size minus 1 must be divisible by a large power of 2). The BN-254 base field does **not** have this property. The ECFFT replaces roots of unity with x-coordinates of points on an elliptic curve, using isogenies in place of the squaring map.

This lets you do O(n log² n) polynomial evaluation and interpolation over **any** prime field.

## Files

| File | Description |
|------|-------------|
| `ecfft_algorithms.py` | Core algorithms: `FFTree`, ENTER, EXIT, EXTEND, DEGREE, and all supporting machinery (field arithmetic, elliptic curve group law, isogenies, binary trees, matrix operations) |
| `ecfft_params_2_18.py` | Curve parameters with a cyclic 2^18 subgroup |
| `ecfft_params_2_19.py` | Curve parameters with a cyclic 2^19 subgroup |
| `ecfft_params_2_20.py` | Curve parameters with a cyclic 2^20 subgroup |
| `demo.ipynb` | Interactive walkthrough of the key ideas |

## Quick start

```python
from ecfft_algorithms import build_fftree, poly_eval
from ecfft_params_2_20 import params

# Build an FFTree with a domain of size 2^5 = 32
tree, leaves = build_fftree(params, log_n=5)

# Coefficients of a polynomial f(x) = 1 + 2x + 3x² + ... + 32x³¹
coeffs = list(range(1, 33))

# ENTER: coefficient representation → evaluation representation  O(n log² n)
evals = tree.enter(coeffs)

# EXIT: evaluation representation → coefficient representation  O(n log² n)
recovered = tree.exit(evals)
assert recovered == coeffs

# DEGREE: compute degree from evaluations  O(n log n)
deg = tree.degree(evals)
assert deg == 31

# EXTEND: given evaluations on one moiety, compute on the other  O(n log n)
domain = tree.eval_domain()
s0_evals = [poly_eval(coeffs[:16], x) for x in domain[0::2]]  # deg < n/2
s1_evals = tree.extend(s0_evals, 'S1')
```

## The FFT analogy

In the classic radix-2 FFT, the key decomposition is:

```
f(x) = f_even(x²) + x · f_odd(x²)
```

The squaring map x ↦ x² is 2-to-1 on roots of unity: it sends ω^i and ω^{i+n/2} to the same point.

The ECFFT replaces the squaring map with ψ(x) = (x − b)²/x, a rational map induced by a "good isogeny" on an elliptic curve. ψ is also 2-to-1 on the evaluation domain. The decomposition becomes:

```
f(x) = u(ψ(x)) + x^{n/2} · v(ψ(x))
```

where u holds the low coefficients and v the high. See `_enter_impl` and `_exit_impl` in the source.

## FRI folding (ECFFT Part II style)

The library provides two styles of FRI-like folding:

### ECFFT2 FRI hash (§12) — use this for FRI protocols

The correct pointwise fold from ECFFT Part II (BSCKL22, Appendix B.2). Each output depends on exactly 2 inputs, enabling O(1) per-query verification.

```python
from ecfft_algorithms import build_fri_domains, ecfri_fold_step, ecfri_fold, ecfri_verify_query
from ecfft_params_2_20 import params

# Build FRI domain layers: L_0 → L_1 → ... → L_k via ψ
layers, psis = build_fri_domains(params, log_n=5)  # L_0 has 32 elements

# Evaluate a polynomial on L_0
from ecfft_algorithms import poly_eval
coeffs = list(range(1, 33))
word = [poly_eval(coeffs, x) for x in layers[0]]

# One FRI fold step (degree bound 32 → 16)
z = 42
folded = ecfri_fold_step(word, layers, round_idx=0, degree_bound=32, z=z)
# folded lives on layers[1] (size 16)

# Verifier checks a single query (O(1)):
j = 5  # query pair index
expected = ecfri_verify_query(layers, 0, 32, j, word[j], word[j + 16], z)
assert expected == folded[j]

# Multi-round fold
folded = ecfri_fold(word, layers, degree_bound=32, challenges=[42, 99, 7])
```

### Global ECFFT decomposition (§11) — for ENTER/EXIT, not FRI

The Part I decomposition `f(x) = u(ψ(x)) + x^{n/2}·v(ψ(x))` uses the FFTree's modular reduction machinery. It is O(n log n) and **not** pointwise — the fold matrix is dense. This is what ENTER/EXIT use internally, but is NOT suitable for FRI verification. See §11 comments in source.

```python
from ecfft_algorithms import build_fftree, ecfft_decompose_step, ecfft_fold_step
from ecfft_params_2_20 import params

tree, _ = build_fftree(params, log_n=5)
evals = tree.enter(list(range(1, 33)))
u_evals, v_evals = ecfft_decompose_step(evals, tree)  # global, O(n log n)
```

## Group-valued BaseFold (§13)

The library includes group-valued FRI folding for the BaseFold protocol from [eprint 2025/1325](https://eprint.iacr.org/2025/1325). This replaces the O(n) MSM in IPA verification with O(λ·log²n) scalar muls.

```python
from ecfft_algorithms import build_fri_domains, basefold_group_fold_step, basefold_verify_query

layers, _ = build_fri_domains(params, log_n=5)
# g_word = list of (x, y) curve points on L_0
# g_folded = basefold_group_fold_step(g_word, layers, 0, 32, z)
```

See §13 in `ecfft_algorithms.py` for the full protocol description.

## Requirements

Python 3.8+. No dependencies.

## Verifying curve parameters

```python
from ecfft_params_2_20 import verify
verify()
```

## License

MIT
