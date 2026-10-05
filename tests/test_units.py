"""Unit tests for the reduced solvers: no reference-package calls.

Everything here runs against the solver's own internals, so the whole file
finishes in seconds. The comparisons against the reference package live in
test_fast_1d.py and test_fast_2d.py, which are marked slow because a single
package build costs between 0.3 and 30 seconds.
"""
import numpy as np
import pytest

from jeans.definitions import GN
from jeans.fast import universal
from jeans.fast.kernels import source_from_grid
from jeans.fast.quadrature import (angular_average_exp, gauss_legendre,
                                   tabulate_baryons)
from jeans.fast.solver import solve_spherical

MD, A_D, B_D = 6e10, 3.0, 0.28
MSTAR, A_H = 5e10, 4.0


def mn_phi(r, th):
    return -GN * MD / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (A_D + np.sqrt(B_D ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2
    )


def hernquist_phi(r):
    return -GN * MSTAR / (r + A_H)


CASES = [(10, 1e12, 10.0), (30, 1e13, 7.0), (3, 1e11, 12.0),
         (20, 5e12, 8.0), (1, 3e10, 15.0)]


# ------------------------------------------------------------------ universal
def test_branch_endpoint_is_the_published_constant():
    R, u = universal.refresh_constants()
    assert R == pytest.approx(1.2615266, abs=1e-6)
    # The location is ill conditioned: g'' at the maximum is -9e-4, so a 1e-9
    # error in g displaces u by 1.5e-3. Converged here to 22.5442.
    assert u == pytest.approx(22.5442, abs=5e-3)


def test_universal_table_is_resolution_converged():
    vals = []
    for n in (60_000, 240_000, 960_000):
        u, phi, eta = universal.table(n, 40.0)
        g = phi - eta
        vals.append(float(np.exp(g.max()) / 3.0))
    assert max(vals) - min(vals) < 1e-9


def test_series_start_matches_the_analytic_expansion():
    u, phi, eta = universal.table()
    k = 40  # u ~ 0.01, deep in the series regime
    assert phi[k] / u[k] ** 2 == pytest.approx(1 / 6, rel=1e-4)
    assert eta[k] / u[k] ** 2 == pytest.approx(1 / 10, rel=1e-4)


def test_matching_function_is_monotonic_on_the_first_branch():
    u, g, _, _ = universal._first_branch()
    assert np.all(np.diff(g) > 0), "R must be invertible by interpolation"


# ------------------------------------------------------------------ quadrature
def test_numba_source_matches_the_numpy_reference():
    nodes = np.linspace(0.0, 10.0, 51)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    gn, gh, w = tabulate_baryons(mn_phi, nodes, half)
    for sig0sq in (1e3, 2.5e4, 1e5):
        ref = -np.log(angular_average_exp(gn, w, sig0sq))
        got = source_from_grid(gn, w, sig0sq)
        assert np.allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_quadrature_converges_spectrally():
    """Gauss-Legendre on a smooth integrand converges geometrically.

    Measured against a 128-node reference for s(r) over the full node range of
    a Miyamoto-Nagai disc, which is the demanding case: 4.8e-6 at 8 nodes,
    2.1e-7 at 12, 1.2e-8 at the default 16, 1.9e-14 by 48. (A published figure
    of 5e-15 at 12 nodes refers to A_L at a single radius, not to s(r) across
    the whole interior.)

    The default of 16 sits two orders below the solver's own RK4 error and four
    below the package's discretisation error, so it is not the limiting term.
    """
    nodes = np.linspace(0.0, 10.0, 51)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    gn, _, w = tabulate_baryons(mn_phi, nodes, half, n=128)
    ref = source_from_grid(gn, w, 2.5e4)

    errs = {}
    for n in (8, 12, 16, 24, 48):
        gn, _, w = tabulate_baryons(mn_phi, nodes, half, n=n)
        s = source_from_grid(gn, w, 2.5e4)
        errs[n] = float(np.max(np.abs(s[1:] / ref[1:] - 1.0)))

    ordered = [errs[n] for n in (8, 12, 16, 24, 48)]
    assert all(b < a for a, b in zip(ordered[:-1], ordered[1:])), errs
    assert errs[16] < 1e-7, f"default node count too coarse: {errs[16]:.2e}"
    assert errs[48] < 1e-12, f"not reaching machine precision: {errs[48]:.2e}"


def test_isotropic_potential_averages_exactly():
    """For a theta-independent Phi_b the average is exact at any node count."""
    nodes = np.linspace(0.0, 10.0, 21)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    gn, _, w = tabulate_baryons(hernquist_phi, nodes, half, n=4)
    s = source_from_grid(gn, w, 2.5e4)
    expect = (hernquist_phi(nodes) - hernquist_phi(0.0)) / 2.5e4
    assert np.allclose(s, expect, rtol=1e-13)


def test_rejects_an_unsupported_signature():
    with pytest.raises(ValueError, match="1 or 2 arguments"):
        tabulate_baryons(lambda r, th, extra: 0.0, np.array([1.0]), np.array([0.5]))
