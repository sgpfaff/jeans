"""Reduced spherical solver against the reference package.

Tolerances are set by the package's own discretisation error, not by a round
number: at its shipped grid it carries ~2.6e-4 relative error in rho, so
agreeing with it to better than that is the most that can be asked. The
convergence test at the end is what actually pins correctness.

Every test here builds a reference profile, so the module is marked slow.
Unit-level tests of the same code live in test_units.py.
"""
import numpy as np
import pytest

from jeans.definitions import GN
from jeans.fast import universal
from jeans.fast.solver import solve_spherical

pytestmark = pytest.mark.slow

MD, A_D, B_D = 6e10, 3.0, 0.28
MSTAR, A_H = 5e10, 4.0


def mn_phi(r, th):
    return -GN * MD / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (A_D + np.sqrt(B_D ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2
    )


def hernquist_phi(r):
    return -GN * MSTAR / (r + A_H)


def _disc(Md, a, b):
    return lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)


CASES = [(10, 1e12, 10.0), (30, 1e13, 7.0), (3, 1e11, 12.0),
         (20, 5e12, 8.0), (1, 3e10, 15.0)]


# ------------------------------------------------------------------ no baryons
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.slow
@pytest.mark.parametrize("r1,M200,c", CASES)
def test_no_baryons_matches_package(r1, M200, c, pkg_spherical, outer_data):
    h = pkg_spherical(r1, M200, c)
    rho1, M1 = outer_data(r1, M200, c)[:2]
    R = solve_spherical(r1, rho1, M1)
    assert R.success
    assert R.r0 == pytest.approx(h.inner.r0, rel=3e-4)
    assert R.sigma0 == pytest.approx(h.inner.sigma0, rel=3e-4)


def test_no_baryons_performs_no_iteration(outer_data):
    rho1, M1 = outer_data(10, 1e12, 10.0)[:2]
    R = solve_spherical(10, rho1, M1)
    assert R.n_residual_evals == 0
    assert R.residual == 0.0


# ------------------------------------------------------------------ existence
def test_existence_criterion_agrees_with_the_solver(outer_data):
    """R >= R_MAX must mean no solution, and conversely."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        r1 = float(rng.uniform(1.0, 30.0))
        M200 = float(10 ** rng.uniform(11.0, 13.0))
        c = float(rng.uniform(5.0, 20.0))
        rho1, M1 = outer_data(r1, M200, c)[:2]
        ratio = universal.matching_ratio_of(r1, rho1, M1)
        R = solve_spherical(r1, rho1, M1)
        assert R.success == (ratio < universal.R_MAX), (
            f"r1={r1} M200={M200:.3g} c={c}: ratio={ratio}, success={R.success}"
        )
        checked += 1
    assert checked == 60


# ------------------------------------------------------------------ baryons
@pytest.mark.parametrize("r1,M200,c,pb,name", [
    (10, 1e12, 10.0, mn_phi, "disc"),
    (20, 1e12, 10.0, mn_phi, "disc-wide"),
    (30, 1e13, 7.0, hernquist_phi, "spheroid"),
    (30, 1e13, 7.0, mn_phi, "disc-big"),
])
def test_with_baryons_matches_package(r1, M200, c, pb, name, pkg_spherical, outer_data):
    h = pkg_spherical(r1, M200, c, Phi_b=pb)
    assert h is not None
    rho1, M1 = outer_data(r1, M200, c, Phi_b=pb)[:2]
    R = solve_spherical(r1, rho1, M1, Phi_b=pb)
    assert R.success and R.residual < 1e-10
    assert R.r0 == pytest.approx(h.inner.r0, rel=5e-4)
    assert R.sigma0 == pytest.approx(h.inner.sigma0, rel=5e-4)


def test_isotropic_fast_path_equals_the_general_path(outer_data):
    """The theta-independent shortcut must not change any answer."""
    r1, M200, c = 30, 1e13, 7.0
    rho1, M1 = outer_data(r1, M200, c, Phi_b=hernquist_phi)[:2]
    fast = solve_spherical(r1, rho1, M1, Phi_b=hernquist_phi)

    # Same potential, declared as two-variable, so the general log-sum-exp runs.
    def as_2d(r, th):
        return hernquist_phi(r) + 0.0 * th

    general = solve_spherical(r1, rho1, M1, Phi_b=as_2d)
    assert fast.r0 == pytest.approx(general.r0, rel=1e-11)
    assert fast.sigma0 == pytest.approx(general.sigma0, rel=1e-11)


# ------------------------------------------------------------------ convergence
def test_self_convergence_in_step_count(outer_data):
    """The solver's own discretisation error must be far below the package gap."""
    r1, M200, c = 10, 1e12, 10.0
    rho1, M1 = outer_data(r1, M200, c, Phi_b=mn_phi)[:2]
    ref = solve_spherical(r1, rho1, M1, Phi_b=mn_phi, n_steps=6400)
    errs = []
    for n in (100, 200, 400, 800):
        R = solve_spherical(r1, rho1, M1, Phi_b=mn_phi, n_steps=n)
        errs.append(abs(R.r0 / ref.r0 - 1.0))
    # Fourth-order RK4 on a smooth integrand: each doubling must gain at least
    # one order of magnitude, and the default (200) must be below 1e-5.
    for a, b in zip(errs[:-1], errs[1:]):
        assert b < a / 8.0, f"convergence stalled: {errs}"
    assert errs[1] < 1e-5, f"n_steps=200 error {errs[1]:.2e} too large"


def test_package_converges_toward_the_fast_solution(pkg_spherical, outer_data):
    """The central validation claim: refining the package's grid closes the gap
    as O(N^-2), with no floor. If the reduction were wrong, a floor would remain."""
    r1, M200, c = 10, 1e12, 10.0
    rho1, M1 = outer_data(r1, M200, c)[:2]
    fast = solve_spherical(r1, rho1, M1)
    errs = []
    for n in (50, 100, 200, 400):
        h = pkg_spherical(r1, M200, c, r_grid=n)
        errs.append(abs(h.inner.r0 / fast.r0 - 1.0))
    orders = [np.log(a / b) / np.log(2.0) for a, b in zip(errs[:-1], errs[1:])]
    assert all(1.8 < o < 2.2 for o in orders), f"orders {orders} from errs {errs}"


def test_schedule_independence_gate_catches_spurious_roots():
    """Past a fold the residual is not a correctness test.

    This configuration is taken from a measured campaign: the 16-stage ramp
    lands on the third sheet at u1 = 2211 with a residual of 2.1e-13, while
    the physical root is at u1 = 11.0 -- an r0 wrong by a factor of 201.

    It used to be a constructed R > R_MAX case guarded by ``if loose.success``,
    which quietly became a no-op once the seed path changed and the ramp began
    failing outright on it. Nothing here is guarded, so if the fixture stops
    exercising the failure the test says so rather than passing.
    """
    r1, rho1, M1 = 14.497472, 6.764072e+05, 2.026979e+10
    pb = _disc(1.669606e+11, 1.431099, 0.614668)

    loose = solve_spherical(r1, rho1, M1, Phi_b=pb, method="ramp", verify=False)
    assert loose.success, "fixture no longer converges"
    assert loose.residual < 1e-10, "fixture no longer has a clean residual"
    assert r1 / loose.r0 > 2000, "fixture no longer lands off-branch"

    gated = solve_spherical(r1, rho1, M1, Phi_b=pb, method="ramp+schedule",
                            verify=True)
    assert not gated.success, "the gate let a spurious root through"
    assert "schedule_dependent" in gated.reason


def test_schedule_independence_accepts_genuine_solutions(outer_data):
    """The gate must not reject answers that are fine."""
    for r1, M200, c in [(10, 1e12, 10.0), (30, 1e13, 7.0), (20, 5e12, 8.0)]:
        rho1, M1 = outer_data(r1, M200, c, 1.0, mn_phi, (0,))[:2]
        g = solve_spherical(r1, rho1, M1, Phi_b=mn_phi,
                            method="ramp+schedule", verify=True)
        u = solve_spherical(r1, rho1, M1, Phi_b=mn_phi,
                            method="ramp", verify=False)
        assert g.success, f"rejected a good solve at r1={r1}: {g.reason}"
        assert g.r0 == pytest.approx(u.r0, rel=1e-12), "verify changed the answer"
