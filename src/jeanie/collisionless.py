"""Collisionless CDM haloes with an isotropic DF, so CDM responds to baryons.

Why this exists
---------------
The isothermal SIDM solve couples baryons to the halo SHAPE, because
rho = rho_0 exp(-Psi/sigma_0^2) depends on the TOTAL potential. The CDM side of
the package has no such coupling: its axis ratio is stipulated constant and
adiabatic contraction only touches the monopole, so a baryon can reshape the
radial profile but never the shape. Any SIDM-vs-CDM shape comparison built on
that is rigged -- measured, not argued: giving CDM one extra shape parameter
destroyed 90-94% of a signature that had looked overwhelming. An automated
search would exploit the same asymmetry far harder than a hand test did, which
is why this module gates the discovery programme.

The construction
----------------
For an ISOTROPIC distribution function f(E), the density is a function of the
relative potential alone,

    rho(x) = 4 pi sqrt(2) int_0^{Psi(x)} f(E) sqrt(Psi(x) - E) dE  ==  F(Psi),

so isodensity surfaces follow isopotential surfaces. That is structurally the
SAME statement as the isothermal case with a different F -- isothermal is the
special case F = exp -- which makes the two models symmetric by construction
rather than by hand.

Why Eddington rather than fitting rho(Psi) to the spherical pairs: with baryons
the total potential runs DEEPER than the seed halo's own Psi_max, and a fitted
F has nothing to say there. The velocity integral does: for Psi > Psi_max the
support of f is simply exhausted and rho grows as sqrt(Psi), with no
extrapolation anywhere. That response is the contraction, and it is also what
makes the halo round up near a concentrated baryonic component.

Why Einasto and not NFW as the seed: NFW's r^-1 cusp makes rho ~ 1/(Psi_max-Psi)
diverge at finite potential, so drho/dPsi carries a non-integrable singularity
at the top of the energy range and the inversion fails inside ~0.1 r_s (we
measured round-trip errors of 500% there, with f(E) going negative). Einasto
has a finite central density, so Psi_max and rho(Psi_max) are both finite and
the DF is well behaved. Simulations prefer Einasto anyway.

Known limitations, in order of how much they should worry you
-------------------------------------------------------------
1. An isotropic DF cannot represent triaxiality. The "sphericalization" this
   reproduces is the halo taking up the total potential's shape, not the
   destruction of box orbits that drives b/a -> 1 in simulations.
2. Real haloes run beta ~ 0.2-0.3 outside the centre. Isotropy will overstate
   the shape response. Anisotropy is the next closure and is a genuine new
   nuisance, not a free parameter we get to pick.
3. The DF is held fixed as the potential deepens. That is the isotropic-DF
   analogue of adiabatic contraction, not a merger history.
"""
import numpy as np

GN = 4.302e-6                      # kpc (km/s)^2 / Msun


# --- the spherical seed halo -------------------------------------------------

def einasto_seed(rho_m2, r_m2, alpha=0.18, n=2000, xmin=1e-5, xmax=1e4):
    """(r, rho, Psi, drho_dPsi) for a spherical Einasto halo.

    Psi = -Phi with Phi(inf) = 0, built from the density by quadrature:
        Psi(r) = G M(r)/r + 4 pi G int_r^inf rho r' dr'
    drho/dPsi is analytic given M(r), since
        drho/dr = -2 rho (r/r_m2)^alpha / r   and   dPsi/dr = -G M(r)/r^2,
    which removes one of the two numerical derivatives the inversion needs.
    """
    r = np.geomspace(xmin, xmax, n) * r_m2
    t = (r / r_m2) ** alpha
    rho = rho_m2 * np.exp(-(2.0 / alpha) * (t - 1.0))

    # M(r): cumulative, seeded with the finite-density core M -> 4/3 pi rho r^3
    integ = 4.0 * np.pi * r ** 2 * rho
    M = np.concatenate([[4.0 / 3.0 * np.pi * r[0] ** 3 * rho[0]],
                        4.0 / 3.0 * np.pi * r[0] ** 3 * rho[0]
                        + np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
    # outer term: int_r^inf rho r' dr', integrated inward from a convergent tail
    g = rho * r
    tail = np.concatenate([np.cumsum((0.5 * (g[1:] + g[:-1])
                                      * np.diff(r))[::-1])[::-1], [0.0]])
    psi = GN * M / r + 4.0 * np.pi * GN * tail

    drho_dpsi = 2.0 * rho * r * t / (GN * M)      # = (drho/dr)/(dPsi/dr)
    return r, rho, psi, drho_dpsi, M


# --- Eddington inversion -----------------------------------------------------

def _energy_grid(psi_max, psi_min, n_lo=500, n_hi=500):
    """E grid resolving BOTH ends.

    A plain geomspace clusters points at small E and leaves the top of the
    energy range -- which sets the inner halo -- unresolved. That alone put the
    inner round trip off by factors of several.
    """
    lo = np.geomspace(psi_min, 0.9 * psi_max, n_lo)
    # Do NOT crowd the top. f(E) varies on a scale of order Psi_max for a
    # cored seed, so nodes within 1e-6 of Psi_max buy no physics and make the
    # log-E spline derivative float64 noise: we measured a spurious
    # dlogf/dlogE ~ 1.3e6 there, which turned the velocity integrand into a
    # spike no fixed quadrature could resolve.
    hi = psi_max - np.geomspace(1e-4 * psi_max, 0.1 * psi_max, n_hi)
    E = np.unique(np.concatenate([lo, hi[hi > 0]]))
    # The inversion works in log E, and the topmost nodes can sit closer than
    # float64 resolves once logged -- strictly increasing in E is not enough.
    keep = np.concatenate([[True], np.diff(np.log(E)) > 1e-12])
    return E[keep]


def eddington_f(psi, drho_dpsi, n_quad=200, n_lo=600, n_hi=300):
    """Isotropic f(E) on a grid of E.

        f(E) = (1/(sqrt(8) pi^2)) dI/dE,
        I(E) = int_0^E (drho/dPsi) dPsi / sqrt(E - Psi)

    The substitution Psi = E(1-t^2) kills the inverse-square-root endpoint
    singularity exactly, leaving I(E) = 2 sqrt(E) int_0^1 (drho/dPsi) dt.
    """
    o = np.argsort(psi)
    psi, drho_dpsi = psi[o], np.asarray(drho_dpsi)[o]
    pos = psi > 0
    psi, drho_dpsi = psi[pos], drho_dpsi[pos]

    E = _energy_grid(psi.max() * (1 - 1e-9), psi.min() * 1.001, n_lo, n_hi)
    t, w = np.polynomial.legendre.leggauss(n_quad)
    t, w = 0.5 * (t + 1.0), 0.5 * w

    I = np.empty_like(E)
    for i, Ei in enumerate(E):
        p = Ei * (1.0 - t ** 2)
        I[i] = 2.0 * np.sqrt(Ei) * np.sum(w * np.interp(p, psi, drho_dpsi))

    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    sp = IUS(np.log(E), I, k=3)
    f = sp.derivative()(np.log(E)) / E / (np.sqrt(8.0) * np.pi ** 2)
    return E, f


def rho_of_psi(psi_eval, E, f, n_quad=200):
    """F(Psi) = 4 pi sqrt(2) int f(E) sqrt(Psi - E) dE, over the support of f.

    Substituting E = Psi - u^2 gives ONE formula for every Psi,

        F = 8 pi sqrt(2) int_{u_lo}^{u_hi} f(Psi - u^2) u^2 du,
        u_lo = sqrt(Psi - min(Psi, E_max)),  u_hi = sqrt(Psi - E_min),

    which absorbs the square-root endpoint exactly rather than straddling it.
    Below E_max the lower limit is zero and this is the usual Psi(1-t^2) form;
    above it the interval is finite and the integrand smooth.

    The two-branch version this replaced was analytically correct -- the
    substitution maps one onto the other -- but disagreed with itself by 47-59%
    at the switch, because sqrt(Psi - E) leaves a square-root endpoint at
    E = E_max that Gauss-Legendre in log E resolves badly. Continuity across
    Psi = E_max is now exact by construction and is asserted in the tests.

    Defined for any Psi >= 0 with no extrapolation of f: above E_max the
    support is simply exhausted and F grows as sqrt(Psi).
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    P = np.atleast_1d(np.asarray(psi_eval, float))
    lE, lf = np.log(E), np.log(np.clip(f, 1e-300, None))
    out = np.zeros_like(P)
    for i, Pi in enumerate(P):
        if Pi <= E[0]:
            continue
        u_lo = np.sqrt(max(Pi - E[-1], 0.0))
        u_hi = np.sqrt(Pi - E[0])
        u = 0.5 * (u_hi - u_lo) * x + 0.5 * (u_hi + u_lo)
        wu = 0.5 * (u_hi - u_lo) * w
        Eq = Pi - u ** 2
        fq = np.exp(np.interp(np.log(np.clip(Eq, E[0], E[-1])), lE, lf))
        fq = np.where((Eq < E[0]) | (Eq > E[-1]), 0.0, fq)
        out[i] = 8.0 * np.pi * np.sqrt(2.0) * np.sum(wu * fq * u ** 2)
    return out
