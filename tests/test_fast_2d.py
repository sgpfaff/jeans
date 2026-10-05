"""Tests for the axisymmetric reduced solver.

Two different tolerances apply here and they should not be confused.

The monopole (r0, sigma0) is solved nonlinearly, so it should agree with the
package to the package's own discretisation error, a few times 1e-4.

The shape phi_L is solved by linear response about the spherical background, so
it carries an additional error quadratic in the halo's own multipoles. At
q0 = 0.8 with no baryons phi_2 reaches 0.18 and the shape agrees to ~3e-3; with
a disc, where phi_2 is an order of magnitude smaller, it agrees to ~5e-4. That
scaling is itself tested below, because it is the signature of the
approximation being what we claim it is.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

import jeans
from jeans.classes import CDM_profile
from jeans.definitions import GN
from jeans.fast.solver2d import solve_axisymmetric

MD, A_D, B_D = 6e10, 3.0, 0.28


def mn_phi(r, th):
    return -GN * MD / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (A_D + np.sqrt(B_D ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2
    )


CASES = [
    ("nobar-q08", 10, 1e12, 10.0, 0.8, None, (0, 2)),
    ("disc-q10", 10, 1e12, 10.0, 1.0, mn_phi, (0, 2)),
    ("disc-q08", 10, 1e12, 10.0, 0.8, mn_phi, (0, 2)),
    ("disc-q09-L4", 10, 1e12, 10.0, 0.9, mn_phi, (0, 2, 4)),
]


@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.parametrize("name,r1,M200,c,q0,pb,Ls", CASES)
def test_monopole_matches_package(name, r1, M200, c, q0, pb, Ls,
                                  pkg_isothermal, outer_data):
    """(r0, sigma0) are solved nonlinearly, so the package's own error bounds us."""
    h = pkg_isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=Ls)
    assert h is not None
    rho1, M1, JL = outer_data(r1, M200, c, q0, pb, Ls)
    R = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=pb)
    assert R.success
    assert R.r0 == pytest.approx(h.inner.r0, rel=5e-4)
    assert R.sigma0 == pytest.approx(h.inner.sigma0, rel=5e-4)


@pytest.mark.parametrize("name,r1,M200,c,q0,pb,Ls", CASES)
def test_shape_matches_package(name, r1, M200, c, q0, pb, Ls,
                               pkg_isothermal, outer_data):
    h = pkg_isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=Ls)
    assert h is not None
    rho1, M1, JL = outer_data(r1, M200, c, q0, pb, Ls)
    R = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=pb)
    rr = 0.999 * r1
    for L in Ls:
        if L == 0:
            continue
        want = h.inner.phi(L, 0, rr)
        got = R.phi_at(L, rr)
        # L=2 is the linear response proper and lands at a few times 1e-5 to
        # 3e-3 depending on how large phi_2 is. L=4 is different in kind: at
        # these parameters phi_2^2 = 2.6e-3 is the same size as phi_4 = 2.5e-3,
        # so the quadratic term the linearisation drops is the *leading*
        # contribution to L=4, not a correction to it. 3.3% is what that costs.
        tol = 5e-3 if L == 2 else 5e-2
        assert got == pytest.approx(want, rel=tol), f"L={L}"


def test_shape_feedback_on_the_monopole_is_not_optional(pkg_isothermal, outer_data):
    """Without it the monopole never feels the shape and r0 is wrong by ~3e-3.

    This pins the reason the solver alternates instead of solving the two
    sectors once each: the package's own r0 moves by +3.6e-3 between L=[0] and
    L=[0,2] for this configuration.
    """
    r1, M200, c, q0, Ls = 10, 1e12, 10.0, 1.0, (0, 2)
    h = pkg_isothermal(r1, M200, c, q0=q0, Phi_b=mn_phi, L_list=Ls)
    rho1, M1, JL = outer_data(r1, M200, c, q0, mn_phi, Ls)

    one = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=mn_phi,
                             n_outer=1)
    many = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=mn_phi,
                              n_outer=3)
    err_one = abs(one.r0 / h.inner.r0 - 1.0)
    err_many = abs(many.r0 / h.inner.r0 - 1.0)
    assert err_one > 1e-3, "expected a single pass to miss the back-reaction"
    assert err_many < 3e-4
    assert err_many < err_one / 10.0


def test_outer_iteration_has_converged_by_three_passes(outer_data):
    r1, M200, c, q0, Ls = 10, 1e12, 10.0, 0.8, (0, 2)
    rho1, M1, JL = outer_data(r1, M200, c, q0, mn_phi, Ls)
    ref = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=mn_phi,
                             n_outer=8)
    three = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=mn_phi,
                               n_outer=3)
    # Measured: 2.9e-3 after one pass, 1.4e-5 after two, 5.8e-8 after three,
    # 2.4e-10 after four. Three is already three orders below the package's own
    # discretisation error, which is where the comparison bottoms out anyway.
    assert three.r0 == pytest.approx(ref.r0, rel=1e-6)
    assert three.phi_at(2, 0.999 * r1) == pytest.approx(
        ref.phi_at(2, 0.999 * r1), rel=1e-6)


def test_linear_response_error_is_quadratic_in_the_multipole(pkg_isothermal, outer_data):
    """The dropped term is quadratic in phi_2, so the *absolute* shape error
    must scale as phi_2^2.

    Note this is the absolute error. The relative error scales as phi_2^1,
    which is why phi_2 agrees better at q0 near 1 even though the absolute
    error is smaller there too. Measured over q0 = 0.95 to 0.70, phi_2 runs
    0.042 to 0.287 and the absolute error runs 3.8e-5 to 1.2e-3, a power of
    1.77.
    """
    r1, M200, c, Ls = 10, 1e12, 10.0, (0, 2)
    amps, errs = [], []
    for q0 in (0.95, 0.85, 0.75, 0.70):
        h = pkg_isothermal(r1, M200, c, q0=q0, L_list=Ls)
        if h is None:
            continue
        rho1, M1, JL = outer_data(r1, M200, c, q0, None, Ls)
        R = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls)
        rr = 0.999 * r1
        want = h.inner.phi(2, 0, rr)
        amps.append(abs(want))
        errs.append(abs(R.phi_at(2, rr) - want))

    assert len(amps) >= 3
    assert amps == sorted(amps), "q0 further from 1 must raise phi_2"
    assert errs == sorted(errs), "and must raise the absolute error"
    power = np.log(errs[-1] / errs[0]) / np.log(amps[-1] / amps[0])
    assert 1.5 < power < 2.5, f"absolute error scales as phi_2^{power:.2f}"


def test_spherical_outer_halo_with_no_baryons_has_no_shape(outer_data):
    """q0=1 and Phi_b=None drives nothing: every L>0 mode must vanish."""
    r1, M200, c, Ls = 10, 1e12, 10.0, (0, 2, 4)
    rho1, M1, JL = outer_data(r1, M200, c, 1.0, None, Ls)
    R = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls)
    assert R.success
    for L in (2, 4):
        assert abs(R.phi_at(L, 0.999 * r1)) < 1e-12, f"spurious L={L} structure"


def test_monopole_reduces_to_the_one_dimensional_solver(outer_data):
    """L_list=[0] must reproduce solve_spherical exactly."""
    from jeans.fast.solver import solve_spherical
    r1, M200, c = 10, 1e12, 10.0
    rho1, M1, JL = outer_data(r1, M200, c, 1.0, mn_phi, (0,))
    a = solve_spherical(r1, rho1, M1, Phi_b=mn_phi)
    b = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=(0,), Phi_b=mn_phi)
    assert b.r0 == pytest.approx(a.r0, rel=1e-12)
    assert b.sigma0 == pytest.approx(a.sigma0, rel=1e-12)


def test_rejects_odd_multipoles():
    with pytest.raises(ValueError, match="odd L"):
        solve_axisymmetric(10.0, 1e7, 1e11, L_list=(0, 3))


def test_requires_monopole_first():
    with pytest.raises(ValueError, match="must start with 0"):
        solve_axisymmetric(10.0, 1e7, 1e11, L_list=(2, 4))
