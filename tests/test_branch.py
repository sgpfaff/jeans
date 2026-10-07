"""Branch selection: the structural facts, and the two fixtures that matter.

The matching problem is multi-valued -- the classical isothermal spiral, folds
at u = 22.544, 244.86, 2634.97 with R -> 1 -- so a root is not an answer until
something says which sheet it is on. These tests pin the three structural facts
the bracketed solver rests on, and then check it on two configurations taken
from a measured campaign rather than constructed by hand:

  * one where the 16-stage ramp returns a root that is 201x wrong with a
    residual of 2.1e-13, and
  * one with R > R_MAX, which the ramp cannot reach at any n_ramp because its
    first stage is the baryon-free problem.

Both are hard-coded with their boundary data so they cannot quietly stop
exercising the failure. The earlier schedule-independence test guarded its
assertions behind ``if loose.success:`` and became a no-op when the seed path
changed; nothing here is guarded.
"""
import numpy as np
import pytest

from jeanie import branch, universal
from jeanie.solver import GN, _Problem, solve_spherical

# ---------------------------------------------------------------- fixtures --
# Measured: the ramp lands on sheet 3 at u1 = 2211 with a machine-precision
# residual, while the physical root is at u1 = 11.0.
SPURIOUS = dict(r1=14.497472, rho1=6.764072e+05, M1=2.026979e+10,
                Md=1.669606e+11, a=1.431099, b=0.614668)
SPURIOUS_R0 = 1.32049          # the physical root
SPURIOUS_RAMP_R0 = 0.00655558  # what the ramp returns

# Measured: R = 1.3033 > R_MAX, so there is no baryon-free solution to start a
# continuation from, but the configuration is perfectly solvable.
ABOVE_RMAX = dict(r1=22.412493, rho1=2.314296e+05, M1=4.267278e+10,
                  Md=1.829392e+11, a=4.385821, b=0.864095)
ABOVE_RMAX_R0 = 1.42635


def disc(Md, a, b):
    return lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)


def hernquist(Mb, ab):
    return lambda r: -GN * Mb / (r + ab)


def problem(d, n_steps=200):
    return _Problem(d["r1"], d["rho1"], d["M1"], n_steps,
                    disc(d["Md"], d["a"], d["b"]), 16, True)


# ------------------------------------------------------- structural facts --
@pytest.mark.parametrize("Phi_b,label", [
    (disc(5e10, 2.5, 0.4), "miyamoto-nagai"),
    (hernquist(4e10, 1.2), "hernquist"),
])
def test_density_residual_is_monotone_in_sigma(Phi_b, label):
    """The inner solve brackets: one sign change, strictly increasing.

    Stated in the solver's own variables, so it does not assume a
    Miyamoto-Nagai disc -- a spherical bulge is checked too.
    """
    P = _Problem(10.0, 1.3e6, 2.4e10, 200, Phi_b, 16, True)
    for u1 in (0.5, 2.0, 8.0, 18.0):
        f = np.array([branch._f_density(P, 10.0 / u1, np.exp(x))
                      for x in np.linspace(0.0, 28.0, 141)])
        ok = np.isfinite(f)
        assert np.all(np.diff(f[ok]) > 0), f"{label}: not monotone at u1={u1}"
        assert int((np.diff(np.sign(f[ok])) != 0).sum()) == 1, \
            f"{label}: not exactly one root at u1={u1}"


@pytest.mark.parametrize("Phi_b,label", [
    (disc(5e10, 2.5, 0.4), "miyamoto-nagai"),
    (hernquist(4e10, 1.2), "hernquist"),
])
def test_matching_residual_is_monotone_in_u1(Phi_b, label):
    """The outer solve brackets, running from -log(3R) at u1 -> 0 upward.

    This is the fact that makes bisection unable to reach another branch. It
    is also a theorem: along the matched curve
    dlogR/dlog u1 = det / (dlog mu / dlog Lam), and both factors are positive
    below the first fold.
    """
    P = _Problem(10.0, 1.3e6, 2.4e10, 200, Phi_b, 16, True)
    us = np.geomspace(1e-3, branch.U_SAFE, 60)
    g = np.array([branch._g(P, 10.0 / u)[0] for u in us])
    assert np.all(np.isfinite(g)), f"{label}: unbracketable somewhere"
    assert np.all(np.diff(g) > 0), f"{label}: matching residual not monotone"
    R = 2.4e10 / (4.0 * np.pi * 10.0 ** 3 * 1.3e6)
    assert g[0] == pytest.approx(-np.log(3.0 * R), abs=1e-3), \
        f"{label}: u1 -> 0 limit is not -log(3R)"


def test_determinant_reproduces_the_sheet_structure():
    """Sheets alternate in sign(det), so the sign alone cannot identify sheet 1.

    Probed at fixed, tiny Lam so that the folds sit at their baryon-free
    values u = 22.544, 244.86, 2634.97 and u1 = 10, 100, 1000 land on sheets
    1, 2 and 3. It has to be done at fixed Lam rather than along the matched
    curve, because there Lam grows roughly as u1^3 at fixed mu, so the baryons
    are never negligible at large u1 however light the disc.
    """
    r1, rho1, Md = 10.0, 1e6, 1e7
    P = _Problem(r1, rho1, 0.9 * 4.0 * np.pi * r1 ** 3 * rho1, 400,
                 disc(Md, 2.5, 0.4), 16, True)
    signs = []
    for u1 in (10.0, 100.0, 1000.0):
        r0 = r1 / u1
        signs.append(np.sign(branch._det(P, r0, GN * Md / (1e-6 * r0))))
    assert signs == [1.0, -1.0, 1.0], f"sheet signs {signs}"


def test_u_safe_sits_inside_the_no_baryon_fold():
    """The fast path must not claim a root at or beyond the first fold.

    U_SAFE is quoted below universal.U_AT_R_MAX, and also below the fold as a
    200-step integration sees it: discretisation moves the no-baryon fold to
    22.544144, which is 6.2e-5 inward of the converged 22.544206.
    """
    assert branch.U_SAFE < universal.U_AT_R_MAX
    assert branch.U_SAFE < 22.544144


# --------------------------------------------------------- the two fixtures --
def test_certificate_rejects_a_measured_spurious_root():
    """The ramp's answer here is 201x wrong with a residual of 2.1e-13.

    It sits at u1 = 2211, on the third sheet, where det > 0 just as it is on
    the physical sheet -- so the determinant sign at the root is not enough and
    the walk down to the regular core is what rejects it.
    """
    d = SPURIOUS
    loose = solve_spherical(d["r1"], d["rho1"], d["M1"],
                            Phi_b=disc(d["Md"], d["a"], d["b"]),
                            method="ramp", verify=False)
    assert loose.success, "fixture no longer converges"
    assert loose.residual < 1e-10, "fixture no longer has a clean residual"
    assert loose.r0 == pytest.approx(SPURIOUS_RAMP_R0, rel=1e-3)

    P = problem(d)
    assert np.sign(branch._det(P, loose.r0, loose.sigma0 ** 2)) > 0, \
        "fixture no longer exercises the same-sign sheet"

    gated = solve_spherical(d["r1"], d["rho1"], d["M1"],
                            Phi_b=disc(d["Md"], d["a"], d["b"]),
                            method="ramp", verify=True)
    assert not gated.success, "the certificate let a spurious root through"
    assert "off_physical_sheet" in gated.reason


def test_bracket_finds_the_physical_root_the_ramp_missed():
    d = SPURIOUS
    got = solve_spherical(d["r1"], d["rho1"], d["M1"],
                          Phi_b=disc(d["Md"], d["a"], d["b"]))
    assert got.success, got.reason
    assert got.r0 == pytest.approx(SPURIOUS_R0, rel=1e-5)
    assert d["r1"] / got.r0 < branch.U_SAFE


def test_bracket_covers_R_above_R_MAX_where_the_ramp_cannot_start():
    """No number of ramp stages helps: stage one is the baryon-free problem."""
    d = ABOVE_RMAX
    R = universal.matching_ratio_of(d["r1"], d["rho1"], d["M1"])
    assert R > universal.R_MAX
    assert universal.solve(d["r1"], d["rho1"], d["M1"], GN=GN) is None

    pb = disc(d["Md"], d["a"], d["b"])
    for n_ramp in (16, 128):
        r = solve_spherical(d["r1"], d["rho1"], d["M1"], Phi_b=pb,
                            method="ramp", n_ramp=n_ramp, verify=False)
        assert not r.success, f"ramp unexpectedly succeeded at n_ramp={n_ramp}"

    got = solve_spherical(d["r1"], d["rho1"], d["M1"], Phi_b=pb)
    assert got.success, got.reason
    assert got.r0 == pytest.approx(ABOVE_RMAX_R0, rel=1e-5)


def test_existence_criterion_agrees_with_the_solve():
    """1/3 < R < R_fold is exactly the condition for a root to exist."""
    for d, expect in ((SPURIOUS, True), (ABOVE_RMAX, True)):
        P = problem(d)
        ok, R, R_fold = branch.exists_with_baryons(P)
        assert ok is expect, f"R={R} R_fold={R_fold}"
        assert R > branch.R_FLOOR
    # and a configuration below the u1 -> 0 floor has no root at all
    r1, rho1 = 10.0, 1e6
    M1 = 0.30 * 4.0 * np.pi * r1 ** 3 * rho1          # R = 0.30 < 1/3
    P = _Problem(r1, rho1, M1, 200, disc(8e11, 2.0, 0.3), 16, True)
    ok, R, _ = branch.exists_with_baryons(P)
    assert not ok and R < branch.R_FLOOR
    assert not solve_spherical(r1, rho1, M1,
                               Phi_b=disc(8e11, 2.0, 0.3)).success


def test_inner_solve_never_returns_a_non_root():
    """Regression: it used to end with ``return 0.5 * (a + b)`` unchecked.

    When false position stagnated that handed back a point which was not a
    root, and the caller read the matching residual off the wrong place on the
    curve -- which put the bracketed solve at u1 = 0.14 when the answer was
    4.56, and reported success.
    """
    d = SPURIOUS
    P = problem(d)
    for u1 in np.geomspace(1e-3, 300.0, 40):
        r0 = d["r1"] / u1
        for guess in (None, 25.0, 2.0):
            ls = branch._sigma_for(P, r0, guess=guess)
            if ls is None:
                continue
            assert abs(branch._f_density(P, r0, np.exp(ls))) < 1e-8, \
                f"non-root returned at u1={u1}, guess={guess}"


def test_bracket_and_ramp_agree_where_the_ramp_is_sound():
    """On ordinary configurations the two must give the same answer."""
    pb = disc(5e10, 2.5, 0.4)
    for r1, rho1, M1 in ((10.0, 1.3e6, 1.306903e10),     # R = 0.80
                         (8.0, 2.5e6, 9.650973e9),      # R = 0.60
                         (12.0, 1.0e6, 2.062895e10),    # R = 0.95
                         (15.0, 9.0e5, 2.671925e10)):   # R = 0.70
        a = solve_spherical(r1, rho1, M1, Phi_b=pb)
        b = solve_spherical(r1, rho1, M1, Phi_b=pb, method="ramp", verify=True)
        assert a.success and b.success, (a.reason, b.reason)
        assert a.r0 == pytest.approx(b.r0, rel=1e-8)
        assert a.sigma0 == pytest.approx(b.sigma0, rel=1e-8)


def test_fold_scan_finds_the_first_fold_when_baryons_move_it_inward():
    """Regression: the scan must not step over a fold that moved inward.

    branch.py records that baryons push the first fold below the no-baryon
    value. Starting the scan at U_SAFE skipped it and returned the second fold
    at ~242, where R_fold is SMALLER -- so exists_with_baryons reported False
    for a configuration solve_spherical solves. A false exclusion is the one
    error an exclusion programme cannot tolerate.
    """
    r1, rho1 = 10.0, 1.0e6
    unit = 4.0 * np.pi * r1 ** 3 * rho1
    m = 0.02 * unit
    pb = lambda r, th: -GN * m / np.maximum(np.asarray(r, float), 0.6 * r1)

    P = _Problem(r1, rho1, 1.0, 400, pb, 16, True)
    u_fold, R_fold = branch.fold_u1(P)
    assert 20.0 < u_fold < 23.0, f"found the wrong fold at u1={u_fold}"
    assert R_fold > 1.2

    M1 = 1.116193 * unit                      # between the two answers
    P2 = _Problem(r1, rho1, M1, 400, pb, 16, True)
    ok, _, _ = branch.exists_with_baryons(P2)
    assert ok is solve_spherical(r1, rho1, M1, Phi_b=pb).success
    assert ok is True
