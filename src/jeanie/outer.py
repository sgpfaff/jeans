"""Fast outer-halo boundary data.

Once the interior solve is reduced, the outer halo is what is left, and almost
all of it is one routine. `compute_potential_moments` evaluates

    J_L = int_{r1}^{inf} dr r^(1-L) rho_L(r),
    rho_L(r) = 2 pi int_0^pi dtheta sin(theta) Z_L(theta) rho(r, theta)

as nested *adaptive* quadrature: an adaptive solve_ivp in r whose integrand is
itself an adaptive quad in theta, repeated over a widening sequence of radial
shells until the tail converges. Measured at 837 ms for q0=0.8 with a disc,
which dominated every two-dimensional end-to-end timing.

Nothing here needs adaptivity. Both integrands are smooth, and the radial one
becomes smooth on a *finite* interval under t = r1/r:

    J_L = r1^(2-L) int_0^1 dt t^(L-3) rho_L(r1/t)

The substitution maps the infinite tail onto t -> 0, where the integrand tends
to t^L / r1^(L+1) for an r^-3 outer profile: finite, and zero for L > 0. So a
fixed Gauss-Legendre rule in t handles the whole range at once, with no shell
sequence and no convergence loop.
"""
import numpy as np

from .quadrature import gauss_legendre, harmonics

__all__ = ["potential_moments", "enclosed_mass", "boundary_data"]

N_R_DEFAULT = 64
N_GL_DEFAULT = 24


def potential_moments(outer_halo, r1, L_list=(0,), n_r=N_R_DEFAULT,
                      n_gl=N_GL_DEFAULT):
    """J_L for each L in L_list, matching CDM_profile.compute_potential_moments.

    Parameters
    ----------
    outer_halo : CDM_profile
        Only its ``rho_sph(r, theta)`` and ``q0`` are used, so adiabatic
        contraction and either halo type come along for free.
    n_r, n_gl : int
        Fixed node counts in t = r1/r and in cos(theta). Both integrands are
        smooth, so convergence is geometric; the defaults sit several orders
        below the package's own discretisation error.

    Notes
    -----
    L > 0 moments vanish identically for a spherical outer halo, and the
    package short-circuits that case. We do the same, both to match it exactly
    and because evaluating the cancellation numerically would return round-off
    rather than zero.
    """
    L_list = list(L_list)
    if any(L % 2 for L in L_list):
        raise ValueError("odd L are not supported (z-symmetry assumed)")

    spherical = getattr(outer_halo, "q0", 1) == 1

    t, wt = gauss_legendre(n_r, half_range=True)     # t in (0, 1), sum(wt) = 1
    x, wx = gauss_legendre(n_gl, half_range=True)    # x = cos(theta) in (0, 1)
    Z = harmonics(L_list, n=n_gl, half_range=True)
    theta = np.arccos(np.clip(x, -1.0, 1.0))

    r = r1 / t                                       # (n_r,)
    # rho_sph takes scalars, so evaluate the grid once and reuse it for every L.
    rho = np.empty((n_r, n_gl))
    for i, ri in enumerate(r):
        for j, tj in enumerate(theta):
            rho[i, j] = outer_halo.rho_sph(float(ri), float(tj))

    out = []
    for k, L in enumerate(L_list):
        if L > 0 and spherical:
            out.append(0.0)
            continue
        # rho_L(r) = 4 pi <Z_L rho>, the factor 4 pi from folding the
        # half-range average back onto the full solid angle.
        rho_L = 4.0 * np.pi * (wx * Z[k] * rho).sum(axis=1)
        out.append(float(r1 ** (2 - L) * (wt * t ** (L - 3) * rho_L).sum()))
    return out


def enclosed_mass(outer_halo, r1, n_r=48, n_gl=16):
    """M(<r1) for a squashed halo, matching CDM_profile.M_encl.

    The package splines a 300-point geomspace table of 4 pi r^2 rho_sph(r) on
    every call, and for a squashed halo each of those rho_sph evaluations runs
    its own angular average: 62 ms per call at q0=0.8, against 0.8 ms when
    spherical. That made it the largest remaining piece of the outer halo once
    the potential moments were dealt with.

    Two changes. The angular average is taken here on fixed nodes from the
    cheap two-argument rho_sph(r, theta) rather than through the package's own
    averaging, and the radial integral uses r = r1 v^2. That substitution is
    what makes a fixed rule work: for an r^-1 central cusp the integrand
    4 pi r^2 rho dr becomes 8 pi r1^3 v^5 rho(r1 v^2) dv, which tends to a
    constant times v^3 as v -> 0 instead of inheriting the cusp, so no inner
    cutoff and no separate tail integral are needed.
    """
    # When the halo is spherical the package already has a cheap path
    # (0.8 ms, against 62 ms squashed), and matching it exactly is worth more
    # than shaving a millisecond.
    if getattr(outer_halo, "q0", 1) == 1:
        return float(outer_halo.M_encl(r1))

    v, wv = gauss_legendre(n_r, half_range=True)
    x, wx = gauss_legendre(n_gl, half_range=True)
    theta = np.arccos(np.clip(x, -1.0, 1.0))

    total = 0.0
    for k, vk in enumerate(v):
        rk = r1 * vk * vk
        rho_avg = 0.0
        for j, tj in enumerate(theta):
            rho_avg += wx[j] * outer_halo.rho_sph(float(rk), float(tj))
        total += wv[k] * vk ** 5 * rho_avg
    return float(8.0 * np.pi * r1 ** 3 * total)


def boundary_data(outer_halo, r1, L_list=(0,), **kw):
    """(rho1, M1, J_L) at the matching radius: everything the solvers need."""
    return (outer_halo.rho_sph_avg(r1),
            enclosed_mass(outer_halo, r1),
            potential_moments(outer_halo, r1, L_list=L_list, **kw))
