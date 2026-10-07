"""Axisymmetric anisotropic DF f(E, L_z): the first-order shape discriminant.

Why this exists
---------------
If rho = F(Psi) with F monotonic then isodensity surfaces ARE isopotential
surfaces, so the axis ratio is independent of F and two models that are both
monotone functions of the same potential have IDENTICAL shapes. Measured, not
argued: collisionless CDM and isothermal SIDM curves coincide exactly in the
same baryonic potential. Shape differences between them are second order,
arising only through the back-reaction of different radial profiles on Psi.

Velocity anisotropy breaks rho = F(Psi) outright, so it is FIRST order. SIDM
thermalises and stays isotropic, and its density follows the isopotentials
exactly; CDM runs beta ~ 0.2-0.3 and does not. That makes anisotropy the
physical origin of any shape discriminant, rather than a nuisance to patch for
fairness -- which is how it was filed until the degeneracy result landed.

The construction
----------------
For f(E, L_z) with E = Psi - v^2/2 and L_z = R v_phi, the meridional velocities
(v_R, v_z) enter only through v_m^2, so they integrate out first:

    rho(R,z) = 2 pi int dv_phi int_0^{Psi - v_phi^2/2} f(eps, R v_phi) deps.

Taking f(E, L_z) = f(E) w(L_z) -- an isotropic DF reweighted by angular
momentum -- the inner integral is a 1D function G(x) = int_0^x f(eps) deps
that is precomputed once, and with v_phi = sqrt(2 Psi) s,

    rho(R,z) = 4 pi sqrt(2 Psi) int_0^1 w(R sqrt(2 Psi) s) G(Psi(1-s^2)) ds

for even w. One quadrature per point. The R-dependence enters ONLY through w,
which is exactly the term that breaks the Psi-only dependence: set w = 1 and
the identity int dv_phi -> 2 sqrt(2(Psi-eps)) recovers
rho = 4 pi sqrt2 int f sqrt(Psi-eps) deps, the isotropic result. That reduction
is asserted in the tests, so the anisotropic machinery cannot silently drift
away from the validated isotropic case.

Sign convention: w suppressing large |L_z| removes circular orbits and biases
the velocity ellipsoid radially (beta > 0, the CDM-like case); w enhancing
large |L_z| biases it tangentially.
"""
import numpy as np

GN = 4.302e-6


def cumulative_f(E, f):
    """G(x) = int_0^x f(eps) deps on the DF's own energy grid.

    f spans many decades, so the running integral is accumulated on the grid
    rather than re-quadratured per evaluation; G is smooth and monotone and
    interpolates well where f does not.
    """
    o = np.argsort(E)
    E, f = E[o], f[o]
    # Piecewise power law, not trapezoid: f spans ~27 decades over <4 decades
    # in E, where a trapezoid rule left the isotropic reduction good to only
    # 6.2e-3. On [E_i, E_i+1] with f ~ E^p the integral is exact.
    lo, hi = f[:-1], f[1:]
    El, Eh = E[:-1], E[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pw = np.log(hi / lo) / np.log(Eh / El)
        seg = np.where(np.abs(pw + 1.0) > 1e-8,
                       (Eh * hi - El * lo) / (pw + 1.0),
                       El * lo * np.log(Eh / El))
    seg = np.where(np.isfinite(seg), seg, 0.5 * (lo + hi) * (Eh - El))
    return E, np.concatenate([[0.0], np.cumsum(seg)])


def gaussian_Lz(L_a):
    """w = exp(-L_z^2 / 2 L_a^2): suppresses circular orbits, radially biased.

    L_a -> inf recovers the isotropic case, which the tests check.
    """
    return lambda Lz: np.exp(-0.5 * (Lz / L_a) ** 2)


def tangential_Lz(L_a):
    """w = 1 + (L_z/L_a)^2: enhances circular orbits, tangentially biased."""
    return lambda Lz: 1.0 + (Lz / L_a) ** 2


def rho_aniso(R, z, psi_of, Egrid, G, w=None, n_quad=120):
    """rho(R, z) for f(E, L_z) = f(E) w(L_z).

    `psi_of(R, z)` returns the relative potential. `Egrid, G` come from
    cumulative_f. `w=None` is the isotropic case and must reproduce the
    isotropic density exactly.
    """
    s, ws = np.polynomial.legendre.leggauss(n_quad)
    s, ws = 0.5 * (s + 1.0), 0.5 * ws                  # on (0, 1)
    Rb, zb = np.broadcast_arrays(np.atleast_1d(np.asarray(R, float)),
                                 np.atleast_1d(np.asarray(z, float)))
    out = np.zeros(Rb.shape, dtype=float)
    for idx in np.ndindex(*Rb.shape):
        Ri, zi = float(Rb[idx]), float(zb[idx])
        P = float(psi_of(Ri, zi))
        if P <= 0:
            continue
        v = np.sqrt(2.0 * P)
        Gq = np.interp(P * (1.0 - s ** 2), Egrid, G, left=0.0, right=G[-1])
        wq = 1.0 if w is None else w(Ri * v * s)
        out[idx] = 4.0 * np.pi * v * np.sum(ws * wq * Gq)
    return out[()] if out.shape == (1,) and np.ndim(R) == 0 else out


def axis_ratio_of(field, r, lo=0.1, hi=4.0, n=60):
    """q such that field(r q^-1/3, equator) == field(r q^2/3, pole).

    Same volume-preserving construction used for the isothermal solver's
    shape, so the two are directly comparable.
    """
    from scipy.optimize import brentq

    def g(q):
        a = float(np.ravel(field(r * q ** (-1 / 3.0), 0.0))[0])   # equator z=0
        b = float(np.ravel(field(0.0, r * q ** (2 / 3.0)))[0])    # pole R=0
        return np.log(a / b)

    glo, ghi = g(lo), g(hi)
    if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0:
        return np.nan                      # genuinely unbracketed, not an error
    return brentq(g, lo, hi, xtol=1e-10)


def cumulative_fx(E, f):
    """H(x) = int_0^x (x - eps) f(eps) deps, the second moment's kernel.

    Needed for the meridional dispersion: with eps = Psi - v_m^2/2 - v_phi^2/2,
    rho <v_m^2> = 4 pi int dv_phi w(R v_phi) H(Psi - v_phi^2/2). Built from the
    same piecewise power-law segments as cumulative_f, since f is equally
    steep here.
    """
    o = np.argsort(E)
    E, f = E[o], f[o]
    lo, hi = f[:-1], f[1:]
    El, Eh = E[:-1], E[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pw = np.log(hi / lo) / np.log(Eh / El)
        m0 = np.where(np.abs(pw + 1.0) > 1e-8,
                      (Eh * hi - El * lo) / (pw + 1.0),
                      El * lo * np.log(Eh / El))
        m1 = np.where(np.abs(pw + 2.0) > 1e-8,
                      (Eh ** 2 * hi - El ** 2 * lo) / (pw + 2.0),
                      El ** 2 * lo * np.log(Eh / El))
    m0 = np.where(np.isfinite(m0), m0, 0.5 * (lo + hi) * (Eh - El))
    m1 = np.where(np.isfinite(m1), m1, 0.5 * (El * lo + Eh * hi) * (Eh - El))
    # H(x) = x * int_0^x f - int_0^x eps f
    I0 = np.concatenate([[0.0], np.cumsum(m0)])
    I1 = np.concatenate([[0.0], np.cumsum(m1)])
    return E, I0, I1


def beta_phi(R, z, psi_of, E, I0, I1, w=None, n_quad=160):
    """Azimuthal anisotropy 1 - <v_phi^2>/<v_R^2> at (R, z).

    For f(E, L_z) the meridional velocities stay isotropic, so sigma_R =
    sigma_z and the anisotropy is purely azimuthal-against-meridional. This is
    what sets how far the density shape departs from the isopotential shape,
    and it is the quantity simulations quote as beta ~ 0.2-0.3 for CDM.
    """
    s, ws = np.polynomial.legendre.leggauss(n_quad)
    s, ws = 0.5 * (s + 1.0), 0.5 * ws
    P = float(psi_of(R, z))
    if P <= 0:
        return np.nan
    v = np.sqrt(2.0 * P)
    x = P * (1.0 - s ** 2)
    G = np.interp(x, E, I0, left=0.0, right=I0[-1])
    H = x * G - np.interp(x, E, I1, left=0.0, right=I1[-1])
    wq = 1.0 if w is None else w(R * v * s)
    num = 2.0 * P * np.sum(ws * wq * s ** 2 * G)      # rho <v_phi^2> / (4 pi v)
    den = np.sum(ws * wq * H)                          # rho <v_m^2> / (8 pi v)
    if den <= 0:
        return np.nan
    return 1.0 - num / den                             # <v_m^2>/2 = <v_R^2>
