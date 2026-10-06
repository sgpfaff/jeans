"""Fast outer-halo boundary data against the reference package."""
import time

import numpy as np
import pytest

from jeans.classes import CDM_profile
from jeans.definitions import GN
from jeanie.outer import boundary_data, potential_moments

pytestmark = pytest.mark.slow

MD, A_D, B_D = 6e10, 3.0, 0.28


def mn_phi(r, th):
    return -GN * MD / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (A_D + np.sqrt(B_D ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2
    )


def hernquist_phi(r):
    return -GN * 5e10 / (r + 4.0)


CASES = [
    ("disc-q08", 1e12, 10.0, 0.8, mn_phi, 10.0, (0, 2)),
    ("disc-q08-L4", 1e12, 10.0, 0.8, mn_phi, 10.0, (0, 2, 4)),
    ("spheroid-q07", 1e13, 7.0, 0.7, hernquist_phi, 30.0, (0, 2)),
    ("nobar-q09", 5e12, 8.0, 0.9, None, 20.0, (0, 2)),
]


@pytest.mark.parametrize("name,M200,c,q0,pb,r1,Ls", CASES)
def test_potential_moments_match_package(name, M200, c, q0, pb, r1, Ls):
    """J_0 to ~3e-7, J_{L>0} to ~1e-8. Both far below anything downstream."""
    o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
    want = o.compute_potential_moments(r1, L_list=list(Ls), M_list=[0] * len(Ls))
    got = potential_moments(CDM_profile(M200, c, q0=q0, Phi_b=pb), r1, L_list=Ls)
    for L, a, b in zip(Ls, want, got):
        assert b == pytest.approx(a, rel=1e-6), f"L={L}"


def test_spherical_outer_halo_gives_exactly_zero_for_L_above_zero():
    """q0=1 must short-circuit, not return round-off.

    Evaluating the cancellation numerically would give ~1e-17 rather than 0,
    and the package returns exact zero, so matching it matters for tests that
    compare the two.
    """
    o = CDM_profile(1e12, 10.0, q0=1.0, Phi_b=mn_phi)
    got = potential_moments(o, 10.0, L_list=(0, 2, 4))
    assert got[1] == 0.0
    assert got[2] == 0.0
    assert got[0] > 0.0


def test_quadrature_is_converged_at_the_defaults():
    """Node counts must be in the flat region, not on a slope."""
    o = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=mn_phi)
    ref = potential_moments(o, 10.0, L_list=(0, 2), n_r=192, n_gl=64)
    for n_r, n_gl in ((48, 16), (64, 24), (96, 32)):
        got = potential_moments(o, 10.0, L_list=(0, 2), n_r=n_r, n_gl=n_gl)
        for L, a, b in zip((0, 2), ref, got):
            assert b == pytest.approx(a, rel=1e-6), f"L={L} at ({n_r},{n_gl})"


def test_rejects_odd_multipoles():
    o = CDM_profile(1e12, 10.0, q0=0.8)
    with pytest.raises(ValueError, match="odd L"):
        potential_moments(o, 10.0, L_list=(0, 3))


def test_boundary_data_bundles_all_three():
    o = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=mn_phi)
    rho1, M1, JL = boundary_data(o, 10.0, L_list=(0, 2))
    assert rho1 > 0 and M1 > 0 and len(JL) == 2
    assert rho1 == pytest.approx(o.rho_sph_avg(10.0), rel=1e-12)
    # M1 now comes from the fast quadrature rather than the package spline,
    # so it agrees to ~6e-11 rather than bit-for-bit.
    assert M1 == pytest.approx(o.M_encl(10.0), rel=1e-8)


def test_is_substantially_faster_than_the_package():
    """The reason this module exists. Measured at 120-145x; assert 20x so the
    test fails on a real regression but not on a busy machine."""
    M200, c, q0, r1, Ls = 1e12, 10.0, 0.8, 10.0, (0, 2)

    o = CDM_profile(M200, c, q0=q0, Phi_b=mn_phi)
    t0 = time.perf_counter()
    o.compute_potential_moments(r1, L_list=list(Ls), M_list=[0] * len(Ls))
    t_pkg = time.perf_counter() - t0

    o2 = CDM_profile(M200, c, q0=q0, Phi_b=mn_phi)
    potential_moments(o2, r1, L_list=Ls)          # warm
    t0 = time.perf_counter()
    potential_moments(o2, r1, L_list=Ls)
    t_fast = time.perf_counter() - t0

    assert t_pkg / t_fast > 20.0, f"only {t_pkg / t_fast:.1f}x"


@pytest.mark.parametrize("name,M200,c,q0,pb,r1,Ls", CASES)
def test_enclosed_mass_matches_package(name, M200, c, q0, pb, r1, Ls):
    """The v^2 substitution should make this essentially exact, not merely close."""
    from jeanie.outer import enclosed_mass
    o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
    assert enclosed_mass(o, r1) == pytest.approx(o.M_encl(r1), rel=1e-8)


def test_enclosed_mass_is_converged_at_the_defaults():
    from jeanie.outer import enclosed_mass
    o = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=mn_phi)
    ref = enclosed_mass(o, 10.0, n_r=256, n_gl=64)
    for n_r, n_gl in ((32, 16), (48, 16), (96, 32)):
        assert enclosed_mass(o, 10.0, n_r=n_r, n_gl=n_gl) == pytest.approx(ref, rel=1e-12)


def test_enclosed_mass_delegates_when_spherical():
    """q0=1 has a cheap package path already; match it exactly rather than
    re-deriving it and introducing a difference."""
    from jeanie.outer import enclosed_mass
    o = CDM_profile(1e12, 10.0, q0=1.0, Phi_b=mn_phi)
    assert enclosed_mass(o, 10.0) == o.M_encl(10.0)


def test_full_boundary_chain_is_much_faster():
    """rho1, M1 and J_L together: measured 97x squashed, 43x spherical."""
    o = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=mn_phi)
    t0 = time.perf_counter()
    o.rho_sph_avg(10.0), o.M_encl(10.0)
    o.compute_potential_moments(10.0, L_list=[0, 2], M_list=[0, 0])
    t_pkg = time.perf_counter() - t0

    o2 = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=mn_phi)
    boundary_data(o2, 10.0, L_list=(0, 2))          # warm
    o2 = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=mn_phi)
    t0 = time.perf_counter()
    boundary_data(o2, 10.0, L_list=(0, 2))
    t_fast = time.perf_counter() - t0

    assert t_pkg / t_fast > 15.0, f"only {t_pkg / t_fast:.1f}x"
