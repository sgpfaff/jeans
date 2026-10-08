"""The multipole expansion, against answers that are exact.

galpy ships no multipole -- SCFPotential is its only basis expansion, and it
does not converge on this profile -- so jeanie carries its own. That means it
has to be validated against closed-form Poisson solutions rather than against
another code, which is what these tests do. The cross-check against agama's
independent Multipole lives in test_export.py.
"""
import numpy as np
import pytest
from scipy.special import eval_legendre

from jeanie.multipole import MultipoleExpansion, GN

A_BREAK = 12.0
C_NORM = 1.0e7


def _single_harmonic(l, k):
    """rho = C r^k P_l(mu) for r <= a and 0 outside, with its exact Phi_l.

    A density that is one harmonic and one power law has a Poisson solution
    in closed form, and cutting it at r = a puts a jump in it -- which is
    exactly the situation at r1. Nothing here is fitted or tolerant: the
    expansion either reproduces an analytic function or it does not.
    """
    def rho(R, z):
        r = np.hypot(R, z)
        mu = np.where(r > 0, z / np.where(r > 0, r, 1.0), 1.0)
        return np.where(r <= A_BREAK, C_NORM * r ** k * eval_legendre(l, mu),
                        0.0)

    def phi_l(r):
        r = np.asarray(r, float)
        pre = -4.0 * np.pi * GN / (2 * l + 1)
        inner = pre * (C_NORM * r ** (k + 2) / (k + l + 3)
                       + r ** l * C_NORM
                       * (A_BREAK ** (k + 2 - l) - r ** (k + 2 - l))
                       / (k + 2 - l))
        outer = pre * C_NORM * A_BREAK ** (k + l + 3) \
            / ((k + l + 3) * r ** (l + 1))
        return np.where(r <= A_BREAK, inner, outer)

    return rho, phi_l


R_PROBE = np.array([0.5, 2.0, 6.0, 11.0, 13.0, 30.0, 100.0])


@pytest.mark.parametrize("l,k", [(2, -1.0), (2, 0.5), (4, -1.0), (6, -2.0)])
def test_harmonics_reproduce_an_exact_poisson_solution(l, k):
    rho, phi_l = _single_harmonic(l, k)
    m = MultipoleExpansion(rho, 1e-3, 1e3, lmax=max(l, 4),
                           breaks=(A_BREAK,), n_r=257, n_theta=40)
    Phi, _ = m.harmonics(R_PROBE)
    err = np.abs(Phi[list(m.ell).index(l)] / phi_l(R_PROBE) - 1)
    assert np.max(err) < 2e-4


def test_the_outer_integral_is_accumulated_inwards():
    """Regression. int_r^inf rho_l r'^(1-l) dr' was built as (total minus a
    running integral from rmin). For l >= 6 the integrand is dominated by the
    inner edge, so that is a subtraction of two numbers twenty orders of
    magnitude apart: l = 6 came out 33% wrong AND STAYED 33% WRONG under
    refinement, which is the only reason it was noticed. Accumulating from
    rmax inwards costs nothing and converges."""
    rho, phi_l = _single_harmonic(6, -2.0)
    prev = None
    for n_r in (129, 257, 513):
        m = MultipoleExpansion(rho, 1e-3, 1e3, lmax=6, breaks=(A_BREAK,),
                               n_r=n_r, n_theta=40)
        Phi, _ = m.harmonics(R_PROBE)
        err = np.max(np.abs(Phi[3] / phi_l(R_PROBE) - 1))
        if prev is not None:
            assert err < prev / 4.0          # fourth order, conservatively
        prev = err
    assert prev < 1e-5


def test_pinning_the_break_is_what_buys_the_accuracy():
    """Not a tuning knob: with the jump smeared across one log interval the
    expansion sits at the per-cent level however fine the grid, which is the
    same place SCF lands. Pinning it is four orders of magnitude."""
    rho, phi_l = _single_harmonic(2, -1.0)
    kw = dict(lmax=4, n_r=513, n_theta=40)
    pinned = MultipoleExpansion(rho, 1e-3, 1e3, breaks=(A_BREAK,), **kw)
    smeared = MultipoleExpansion(rho, 1e-3, 1e3, breaks=(), **kw)
    ex = phi_l(R_PROBE)
    e_pin = np.max(np.abs(pinned.harmonics(R_PROBE)[0][1] / ex - 1))
    e_smr = np.max(np.abs(smeared.harmonics(R_PROBE)[0][1] / ex - 1))
    assert e_pin < 1e-5
    assert e_smr > 100 * e_pin


def test_hernquist_monopole_is_exact():
    M, a = 1.0e11, 5.0
    def rho(R, z):
        r = np.hypot(R, z)
        return (M * a / (2 * np.pi)) / (r * (r + a) ** 3)
    m = MultipoleExpansion(rho, 1e-4, 1e5, lmax=4, n_r=513)
    r = np.geomspace(1e-2, 1e3, 12)
    assert np.max(np.abs(m.potential(r, 0.0) / (-GN * M / (r + a)) - 1)) < 1e-6
    FR, _ = m.forces(r, 0.0)
    assert np.max(np.abs(FR / (-GN * M / (r + a) ** 2) - 1)) < 1e-4
    assert abs(m.M_total / M - 1) < 1e-5


def test_miyamoto_nagai_converges_in_lmax():
    """The angular machinery, on a flattened density with a closed-form
    potential. A razor-ish disc is the worst case for a multipole, so the
    numbers are modest -- the point is that they fall with lmax."""
    Md, a, b = 5.0e10, 1.0, 0.8
    def rho(R, z):
        zb = np.sqrt(z ** 2 + b ** 2)
        num = a * R ** 2 + (a + 3 * zb) * (a + zb) ** 2
        return b ** 2 * Md / (4 * np.pi) * num \
            / ((R ** 2 + (a + zb) ** 2) ** 2.5 * zb ** 3)
    exact = lambda R, z: -GN * Md / np.sqrt(
        R ** 2 + (a + np.sqrt(z ** 2 + b ** 2)) ** 2)
    pts = [(2., 0.), (2., 1.), (5., 0.), (5., 5.), (10., 3.), (30., 0.)]
    errs = []
    for lmax in (4, 16):
        m = MultipoleExpansion(rho, 1e-3, 1e4, lmax=lmax, n_r=513, n_theta=48)
        errs.append(max(abs(float(m.potential(R, z)) / exact(R, z) - 1)
                        for R, z in pts))
    assert errs[0] < 1e-2
    assert errs[1] < errs[0] / 5.0


def test_the_force_is_the_derivative_of_the_interpolated_potential():
    """Not merely of the true potential -- of the one that is represented.
    That is what makes an orbit conserve energy, and it is why dPhi/dr comes
    from differentiating the Hermite interpolant rather than from a second
    interpolant through the node derivatives."""
    M, a = 1.0e11, 5.0
    rho = lambda R, z: (M * a / (2 * np.pi)) / (
        np.hypot(R, z) * (np.hypot(R, z) + a) ** 3)
    m = MultipoleExpansion(rho, 1e-3, 1e4, lmax=4, n_r=257)
    h = 1e-5
    for R, z in [(2.0, 1.0), (7.0, 3.0), (20.0, 0.5)]:
        fdR = -(float(m.potential(R + h, z))
                - float(m.potential(R - h, z))) / (2 * h)
        fdz = -(float(m.potential(R, z + h))
                - float(m.potential(R, z - h))) / (2 * h)
        FR, Fz = m.forces(R, z)
        assert abs(float(FR) / fdR - 1) < 1e-6
        assert abs(float(Fz) / fdz - 1) < 1e-6
