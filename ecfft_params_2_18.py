"""
ECFFT Parameters — Curve with 2^18 cyclic subgroup over BN-254 base field
===========================================================================

This file provides concrete curve parameters for an elliptic curve  E  over the
BN-254 base field  Fq  (q = 21888242871839275222246405745257275088696311157297823662689037894645226208583)
whose group  E(Fq)  contains a cyclic subgroup of order  2^18 = 262144.

The curve is a "Good Curve" in the sense of ECFFT Part II:

    E: y² = x³ + a·x² + B·x

with the 2-torsion point  T = (0, 0)  in its kernel, enabling a chain of
18 good isogenies that drive the ECFFT recursion.


Curve parameters
-----------------

Found by random search over BN-254 base field.

    a  = 19278762604304102184610580119031188699044367773724953696453398365456011135758
    B  = 5057265613299442410214857458322430525862354560484657463708792002756627573613

    Generator point (order 2^18):
    G  = (15269060823305261951304673662723140852191989236410665347887418708344110802446,
          9348477176575789887863776815485170254906958494099604420587794525131513185024)


Verification
-------------

To verify these parameters:

    >>> from ecfft_params_2_18 import params, verify
    >>> verify()
    True

To run the ECFFT demo:

    >>> from ecfft_algorithms import demo
    >>> from ecfft_params_2_18 import params
    >>> demo(params, size=8)
"""

from ecfft_algorithms import (
    q, fadd, fsub, fmul, fdiv, fneg, finv, fsqrt, fpow,
    GoodCurve, Point, RationalMap,
    good_isogeny, apply_isogeny, build_isogeny_chain,
    build_evaluation_domain, split_domain_with_psi,
    build_fftree, FFTree, lagrange_interpolate,
    poly_eval, demo
)

# ---------------------------------------------------------------------------
# Curve Parameters
# ---------------------------------------------------------------------------

params = {
    'a'  : 19278762604304102184610580119031188699044367773724953696453398365456011135758,
    'bb' : 5057265613299442410214857458322430525862354560484657463708792002756627573613,
    'gx' : 15269060823305261951304673662723140852191989236410665347887418708344110802446,
    'gy' : 9348477176575789887863776815485170254906958494099604420587794525131513185024,
    'k'  : 18,
}


def verify():
    """
    Verify that the parameters are correct:
      1. (gx, gy) lies on the curve  y² = x³ + a·x² + B·x
      2. B is a quadratic residue (b² = B)
      3. The generator has order exactly 2^18
      4. The 2-Sylow subgroup is cyclic (discriminant a²-4B is a QNR)
    """
    a, bb, gx, gy, k = params['a'], params['bb'], params['gx'], params['gy'], params['k']

    # 1. Point on curve
    curve = GoodCurve(a, bb)
    assert curve.contains(gx, gy), "Generator not on curve"
    print(f"  ✓ Point ({gx}, {gy}) lies on E")

    gen = Point(gx, gy, curve)

    # 2. B is a QR
    b = fsqrt(bb)
    assert b is not None, "B is not a quadratic residue"
    print(f"  ✓ B = b² where b = {b}")

    # 3. Generator has order 2^k
    p = gen.scalar_mul(2 ** k)
    assert p.is_infinity(), f"2^{k} * G ≠ ∞"
    print(f"  ✓ 2^{k} · G = ∞")

    # Order is exactly 2^k (not less)
    p_half = gen.scalar_mul(2 ** (k - 1))
    assert not p_half.is_infinity(), f"2^{k-1} * G = ∞ (order < 2^{k})"
    print(f"  ✓ 2^{k-1} · G ≠ ∞  (order is exactly 2^{k})")

    # 4. Cyclic 2-Sylow: discriminant a²-4B must NOT be a QR
    disc = fsub(fmul(a, a), fmul(4, bb))
    assert fsqrt(disc) is None, "Discriminant is a QR — 2-Sylow is not cyclic"
    print(f"  ✓ Discriminant a²−4B is a QNR → cyclic 2-Sylow subgroup")

    print(f"\n  All verifications passed for 2^{k} curve.")
    return True


def build_psi_maps(depth=4):
    """
    Build the first `depth` ψ maps (rational maps on ℙ¹) for this curve.

    Each ψ_i is a 2-to-1 map:
        ψ_i(x) = (x - b_i)² / x

    where b_i is the square root of B_i on the i-th curve in the isogeny chain.

    Returns
    -------
    psi_maps : list of RationalMap
    curves   : list of GoodCurve
    """
    a, bb, gx, gy, k = params['a'], params['bb'], params['gx'], params['gy'], params['k']
    curve = GoodCurve(a, bb)
    gen = Point(gx, gy, curve)

    # Scale generator to order 2^depth
    scaled = gen.scalar_mul(2 ** (k - depth))
    psi_maps, curves, h_maps = build_isogeny_chain(scaled, depth)

    return psi_maps, curves


def print_psi_maps(depth=4):
    """Print the ψ maps in human-readable form."""
    psi_maps, curves = build_psi_maps(depth)

    print(f"\nIsogeny chain for 2^{params['k']} curve (first {depth} levels):")
    print(f"{'='*72}")

    for i, (psi, curve) in enumerate(zip(psi_maps, curves)):
        b = curve.b
        print(f"\n  Level {i}:")
        print(f"    Curve E_{i}: y² = x³ + {curve.a}·x² + {curve.bb}·x")
        print(f"    b_{i} = √B_{i} = {b}")
        print(f"    ψ_{i}(x) = (x - {b})² / x")
        print(f"             = (x² - {fmul(2, b)}·x + {fmul(b, b)}) / x")

    print(f"\n  Final curve E_{depth}: y² = x³ + {curves[-1].a}·x² + {curves[-1].bb}·x")


# ---------------------------------------------------------------------------
# Quick demo on small domain
# ---------------------------------------------------------------------------

def small_demo(size=8):
    """
    Run a small ECFFT demo (default size 8 = 2³).

    This exercises:
      - Isogeny chain construction (3 levels)
      - Evaluation domain construction
      - ψ map 2-to-1 verification
      - ENTER (coefficient → evaluation)
      - EXIT (evaluation → coefficient)
      - Round-trip verification
    """
    return demo(params, size=size)


# ---------------------------------------------------------------------------
# Demonstrate the ψ maps as 2-to-1 maps on ℙ¹
# ---------------------------------------------------------------------------

def demonstrate_psi_two_to_one(size=8):
    """
    Show concretely that each ψ_i is 2-to-1 on the evaluation domain.

    For a domain of size `size = 2^d`, we build d isogenies and show that
    ψ₀ maps the domain 2-to-1 onto a set of size `size/2`, ψ₁ maps that
    set 2-to-1 onto a set of size `size/4`, and so on.
    """
    a, bb, gx, gy, k = params['a'], params['bb'], params['gx'], params['gy'], params['k']
    curve = GoodCurve(a, bb)
    gen = Point(gx, gy, curve)

    log_size = size.bit_length() - 1
    scaled = gen.scalar_mul(2 ** (k - log_size))
    psi_maps, curves, _ = build_isogeny_chain(scaled, log_size)

    # Use 2*gen as coset offset
    coset_offset = gen.double()
    domain = build_evaluation_domain(scaled, coset_offset, size)

    print(f"\nDemonstrating ψ maps as 2-to-1 maps on ℙ¹ (domain size {size}):")
    print(f"{'='*72}")

    current_domain = domain
    for i in range(log_size):
        psi = psi_maps[i]
        n = len(current_domain)
        s0, s1, images = split_domain_with_psi(current_domain, psi)

        print(f"\n  ψ_{i}: domain of size {n} → images of size {len(images)}")
        print(f"    Moiety S₀: {s0[:3]}{'...' if len(s0)>3 else ''}")
        print(f"    Moiety S₁: {s1[:3]}{'...' if len(s1)>3 else ''}")
        print(f"    Images:    {images[:3]}{'...' if len(images)>3 else ''}")

        # Verify each pair maps to the same image
        for j in range(min(3, len(s0))):
            img0 = psi(s0[j])
            img1 = psi(s1[j])
            assert img0 == img1
            print(f"    ψ_{i}(S₀[{j}]) = ψ_{i}(S₁[{j}]) = {img0}")

        current_domain = images

    print(f"\n  Final domain (size {len(current_domain)}): {current_domain}")
    print(f"  ✓ All ψ maps verified as 2-to-1")


if __name__ == "__main__":
    print("Verifying curve parameters...")
    verify()
    print()

    print("Printing ψ maps (first 4 levels)...")
    print_psi_maps(depth=4)
    print()

    print("Demonstrating ψ maps as 2-to-1 on ℙ¹...")
    demonstrate_psi_two_to_one(size=8)
    print()

    print("Running small ECFFT demo (size=8)...")
    small_demo(size=8)
