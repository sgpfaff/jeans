import numpy as np
import pytest

from jeanie.collisionless import einasto_seed, eddington_f, rho_of_psi
from jeanie.anisotropic import (cumulative_f, cumulative_fx, rho_aniso,
                                gaussian_Lz, axis_ratio_of, beta_phi)

GN = 4.302e-6
SEED = dict(rho_m2=1.0e7, r_m2=20.0, alpha=0.18)


@pytest.fixture(scope="module")
def df():
    r, rho, psi, drho, M = einasto_seed(**SEED)
    E, f = eddington_f(psi, drho)
    Eg, G = cumulative_f(E, f)
    E2, I0, I1 = cumulative_fx(E, f)
    return r, rho, psi, E, f, Eg, G, E2, I0, I1


def _psi_tot(r, psi, Md=6.0e10, ad=3.0, bd=0.28):
    def p(R, z):
        return (np.interp(np.hypot(R, z), r, psi)
                + GN * Md / np.sqrt(R ** 2 + (ad + np.sqrt(bd ** 2 + z ** 2)) ** 2))
    return p


def test_isotropic_limit_reproduces_the_F_of_psi_density(df):
    """w = 1 must give back rho_of_psi exactly.

    The two routes are independent -- a velocity integral against a direct
    evaluation of F(Psi) -- so this is what stops the anisotropic machinery
    drifting away from the validated isotropic case.
    """
    r, rho, psi, E, f, Eg, G, *_ = df
    rt = np.geomspace(1.0, 200.0, 12)
    psi_sph = lambda R, z: np.interp(np.hypot(R, z), r, psi)
    a = rho_aniso(rt, np.zeros_like(rt), psi_sph, Eg, G)
    b = rho_of_psi(np.interp(rt, r, psi), E, f)
    assert np.max(np.abs(a / b - 1.0)) < 5e-3


def test_isotropic_density_shape_equals_the_isopotential_shape(df):
    """The first-order degeneracy, checked where it would hurt most.

    If rho = F(Psi) with F monotonic then isodensity surfaces ARE isopotential
    surfaces, so the axis ratio cannot depend on F. This is the reason shape
    cannot discriminate SIDM from CDM at first order, and it is asserted here
    in a flattened baryonic potential rather than argued.
    """
    r, rho, psi, E, f, Eg, G, *_ = df
    pt = _psi_tot(r, psi)
    for rr in (4.0, 8.0, 20.0):
        q_psi = axis_ratio_of(lambda R, z: -pt(R, z), rr)
        q_rho = axis_ratio_of(lambda R, z: rho_aniso(R, z, pt, Eg, G), rr)
        assert abs(q_rho - q_psi) < 2e-3, f"at r={rr}: {q_rho} vs {q_psi}"


def test_anisotropy_breaks_the_degeneracy_and_scales_with_beta(df):
    """Anisotropy is the FIRST-order shape discriminant.

    f(E, L_z) makes rho depend on cylindrical R as well as Psi, so the
    isodensity surfaces leave the isopotential ones. Measured departure
    q_rho - q_Psi ~ 0.27 beta, i.e. ~0.08 at the beta ~ 0.3 simulations give
    for CDM -- four times a plausible measurement error, and unlike the
    saturating shape response it does not die at large cross-section.
    """
    r, rho, psi, E, f, Eg, G, E2, I0, I1 = df
    pt = _psi_tot(r, psi)
    q_psi = axis_ratio_of(lambda R, z: -pt(R, z), 8.0)
    prev_b = prev_d = -np.inf
    for La in (8000.0, 5000.0, 3500.0, 2500.0):
        w = gaussian_Lz(La)
        b = beta_phi(8.0, 0.0, pt, E2, I0, I1, w=w)
        q = axis_ratio_of(lambda R, z: rho_aniso(R, z, pt, Eg, G, w=w), 8.0)
        d = q - q_psi
        assert b > prev_b and d > prev_d        # both grow as L_a tightens
        assert d > 0                            # radial bias -> prolate
        prev_b, prev_d = b, d
    assert 0.2 < d / b < 0.4                    # the ~0.27 beta scaling


def test_beta_vanishes_for_the_isotropic_df(df):
    """A sanity check on the moments themselves, not on the shape."""
    r, rho, psi, E, f, Eg, G, E2, I0, I1 = df
    pt = _psi_tot(r, psi)
    assert abs(beta_phi(8.0, 0.0, pt, E2, I0, I1)) < 0.02
