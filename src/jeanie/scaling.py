"""Scale-free Einasto DF, so Eddington inversion runs once instead of per halo.

The discovery search needs 10^4-10^6 haloes. Eddington inversion costs a few
hundred ms, which would dominate everything; but for fixed shape index alpha
the Einasto DF is UNIVERSAL once stripped of its scales, so the inversion is
done a single time per alpha and rescaled for free.

Dimensional analysis. With rho = rho_m2 * rho~(x), x = r/r_m2:

    Psi  = (G rho_m2 r_m2^2) Psi~(x)        [velocity^2]
    E    = (G rho_m2 r_m2^2) E~
    f    = rho_m2 (G rho_m2 r_m2^2)^(-3/2) f~(E~)
    L_z  = r_m2 sqrt(G rho_m2 r_m2^2) L~

so one dimensionless table (E~, f~) serves every (rho_m2, r_m2) at fixed
alpha. The round-trip test asserts the rescaling against a directly inverted
halo rather than trusting the algebra.
"""
import numpy as np

from .collisionless import einasto_seed, eddington_f, GN
from .anisotropic import cumulative_f, cumulative_fx

_CACHE = {}


def scaled_df(alpha=0.18, n=2000, n_lo=600, n_hi=300):
    """Dimensionless (x, rho~, Psi~, E~, f~, G~, I0~, I1~) for an Einasto halo.

    Computed with rho_m2 = r_m2 = 1, which makes G rho_m2 r_m2^2 = G and the
    unit conversions below pure powers.
    """
    key = (alpha, n, n_lo, n_hi)
    if key in _CACHE:
        return _CACHE[key]
    r, rho, psi, drho, M = einasto_seed(1.0, 1.0, alpha, n=n)
    E, f = eddington_f(psi, drho, n_lo=n_lo, n_hi=n_hi)
    Eg, G = cumulative_f(E, f)
    E2, I0, I1 = cumulative_fx(E, f)
    _CACHE[key] = dict(x=r, rho=rho, psi=psi, E=E, f=f, Eg=Eg, G=G,
                       E2=E2, I0=I0, I1=I1)
    return _CACHE[key]


def units(rho_m2, r_m2):
    """Scale factors RELATIVE TO THE TABLE, which already carries G.

    The table is built with rho_m2 = r_m2 = 1 and the real G, so its potential
    unit is G*1*1 and the ratio to physical is rho_m2 r_m2^2 with no further
    factor of G. Including one here double-counts it -- caught by the
    round-trip test, which gave |ratio-1| = 1.0 exactly rather than ~1e-10.
    """
    u_psi = rho_m2 * r_m2 ** 2
    return dict(rho=rho_m2, psi=u_psi, r=r_m2,
                L=r_m2 * np.sqrt(u_psi),
                f=rho_m2 * u_psi ** -1.5,
                G=rho_m2 * u_psi ** -0.5,        # G = int f dE  ->  f * psi
                I1=rho_m2 * u_psi ** 0.5)        # I1 = int eps f dE


def rescale(tab, rho_m2, r_m2):
    """Put a scaled DF table into physical units for one halo.

    Returns the pieces rho_aniso and beta_phi need: (r, psi, Eg, G, E2, I0,
    I1), all in kpc, (km/s)^2 and Msun/kpc^3.
    """
    u = units(rho_m2, r_m2)
    return dict(r=tab["x"] * u["r"],
                psi=tab["psi"] * u["psi"],
                Eg=tab["Eg"] * u["psi"], G=tab["G"] * u["G"],
                E2=tab["E2"] * u["psi"],
                I0=tab["I0"] * u["G"], I1=tab["I1"] * u["I1"])
