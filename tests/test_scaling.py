import numpy as np

from jeanie.collisionless import einasto_seed, eddington_f
from jeanie.anisotropic import cumulative_f, rho_aniso, gaussian_Lz
from jeanie.scaling import scaled_df, rescale

ALPHA = 0.18


def test_rescaled_df_matches_a_directly_inverted_halo():
    """One inversion, rescaled, must equal inverting each halo separately.

    This is what makes the discovery search affordable -- 15 us per halo
    against 22 ms -- so it has to be exact, not approximate. The scaled table
    is built with rho_m2 = r_m2 = 1 and the REAL G, so its potential unit
    already carries G; including another factor in the rescaling gave
    |ratio-1| = 1.0 exactly, which is what this test was written to catch.
    """
    rho_m2, r_m2 = 3.7e6, 31.0                 # not the reference halo
    sc = rescale(scaled_df(ALPHA), rho_m2, r_m2)
    r, rho, psi, drho, M = einasto_seed(rho_m2, r_m2, ALPHA)
    E, f = eddington_f(psi, drho)
    Eg, G = cumulative_f(E, f)

    assert np.max(np.abs(np.interp(r, sc["r"], sc["psi"]) / psi - 1)) < 1e-10

    ps = lambda R, z: np.interp(np.hypot(R, z), r, psi)
    rt = np.geomspace(2.0, 300.0, 10)
    for w in (None, gaussian_Lz(4000.0)):
        a = rho_aniso(rt, np.zeros_like(rt), ps, sc["Eg"], sc["G"], w=w)
        b = rho_aniso(rt, np.zeros_like(rt), ps, Eg, G, w=w)
        assert np.max(np.abs(a / b - 1.0)) < 1e-9


def test_scaled_table_is_cached():
    """Repeated calls must not re-invert; the whole point is one inversion."""
    a = scaled_df(ALPHA)
    assert scaled_df(ALPHA) is a
