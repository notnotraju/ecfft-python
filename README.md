# ecfft-python

Pure-Python implementation of the **Elliptic Curve Fast Fourier Transform** (ECFFT) over the BN-254 base field, organized around the **group-valued BaseFold** protocol from [eprint 2025/1325](https://eprint.iacr.org/2025/1325).

This is a literate, dependency-free implementation based on:

- [ECFFT Part I: Fast Polynomial Algorithms over all Finite Fields](https://arxiv.org/pdf/2107.08473.pdf) (Ben-Sasson, Carmon, Frankel, Kopparty)
- [ECFFT Part II](https://www.math.toronto.edu/swastik/ECFFT2.pdf) (Ben-Sasson, Carmon, Kopparty, Levit)
- [Revisiting the IPA-sumcheck connection](https://eprint.iacr.org/2025/1325) (Eagen, Gabizon)

Rust reference: [andrewmilson/ecfft](https://github.com/andrewmilson/ecfft).

## The problem

In IPA verification, the "decide" step computes $G(r) = \sum_i s_i \cdot G_i$ — an $O(n)$ MSM that dominates recursive verification cost. Group BaseFold replaces this with a FRI-like protocol over **group elements**, reducing the verifier to $O(\lambda \cdot \log^2 n)$ scalar multiplications via Merkle-committed fold oracles and random spot-checks.

The ECFFT is what makes this work over BN-254: the base field has no roots of unity of large 2-power order, so the standard FFT doesn't apply. The ECFFT replaces roots of unity with x-coordinates of elliptic curve points, and the squaring map $x \mapsto x^2$ with a rational map $\psi(x) = (x - b)^2/x$ induced by a good isogeny.

## Files

| File | Purpose |
|------|---------|
| `ecfft_algorithms.py` | **Core**: field arithmetic, curves, isogenies, FRI domains, ECFFT2 pointwise fold, group-valued BaseFold |
| `ecfft_fftree.py` | **General ECFFT**: FFTree with ENTER/EXIT/EXTEND/DEGREE, global Part I decomposition |
| `ecfft_params_2_18.py` | Curve parameters with a cyclic $2^{18}$ subgroup |
| `ecfft_params_2_19.py` | Curve parameters with a cyclic $2^{19}$ subgroup |
| `ecfft_params_2_20.py` | Curve parameters with a cyclic $2^{20}$ subgroup |
| `demo.ipynb` | Interactive walkthrough |

## Quick start: Group BaseFold

```python
from ecfft_algorithms import build_fri_domains, basefold_group_fold_step, basefold_verify_query
from ecfft_params_2_20 import params

# Build FRI domain layers: L_0 -> L_1 -> ... -> L_k via psi
layers, psis = build_fri_domains(params, log_n=5)  # L_0 has 32 elements

# g_word = list of (x, y) group elements on L_0
# g_folded = basefold_group_fold_step(g_word, layers, round_idx=0, degree_bound=32, z=42)
# expected = basefold_verify_query(layers, 0, 32, j, g_word[j], g_word[j+16], z=42)
# assert expected == g_folded[j]  # O(1) per-query verification
```

## Quick start: Scalar FRI fold

```python
from ecfft_algorithms import build_fri_domains, ecfri_fold_step, ecfri_fold, ecfri_verify_query, poly_eval
from ecfft_params_2_20 import params

layers, psis = build_fri_domains(params, log_n=5)

# Evaluate a polynomial on L_0
coeffs = list(range(1, 33))
word = [poly_eval(coeffs, x) for x in layers[0]]

# One FRI fold step (degree bound 32 -> 16)
z = 42
folded = ecfri_fold_step(word, layers, round_idx=0, degree_bound=32, z=z)

# Verifier checks a single query (O(1)):
j = 5
expected = ecfri_verify_query(layers, 0, 32, j, word[j], word[j + 16], z)
assert expected == folded[j]

# Multi-round fold
folded = ecfri_fold(word, layers, degree_bound=32, challenges=[42, 99, 7])
```

## Quick start: General ECFFT (FFTree)

For polynomial evaluation and interpolation without roots of unity:

```python
from ecfft_fftree import build_fftree
from ecfft_params_2_20 import params

tree, leaves = build_fftree(params, log_n=5)

coeffs = list(range(1, 33))
evals = tree.enter(coeffs)        # O(n log^2 n) evaluation
recovered = tree.exit(evals)      # O(n log^2 n) interpolation
assert recovered == coeffs

deg = tree.degree(evals)           # O(n log n) degree computation
assert deg == 31
```

## How the ECFFT fold works

In the classic FFT, the squaring map $x \mapsto x^2$ is 2-to-1 on roots of unity, and FRI folding exploits this: $f(\omega^i)$ and $f(-\omega^i)$ share the same even/odd decomposition values, so folding is a pointwise $2 \times 2$ operation.

The ECFFT replaces the squaring map with $\psi(x) = (x - b)^2/x$, which is 2-to-1 on evaluation domains built from curve points. For a pair $(s_0, s_1)$ with $\psi(s_0) = \psi(s_1)$ and degree bound $d$:

$$e = d/2 - 1$$

$$a = P(s_0) / s_0^e, \quad b = P(s_1) / s_1^e$$

$$\text{slope} = \frac{b - a}{s_1 - s_0}$$

$$H_z[P](t) = a + \text{slope} \cdot (z - s_0) = P_0(t) + z \cdot P_1(t)$$

This is **pointwise**: each output depends on exactly 2 inputs. This is what makes FRI verification $O(1)$ per query — and what makes group BaseFold possible, since scalar-multiplying a group element is cheap but a dense matrix-vector product over group elements is not.

## Requirements

Python 3.8+. No dependencies.

## Verifying curve parameters

```python
from ecfft_params_2_20 import verify
verify()
```

## License

MIT
