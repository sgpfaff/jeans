"""Outer-halo boundary data in closed form, differentiable in the halo parameters.

:mod:`jeans.fast.outer` made the boundary fast by replacing nested adaptive
quadrature with fixed nodes, but it still calls the package's ``rho_sph`` on
those nodes, which is a host function. That blocks ``vmap`` and, more
importantly, blocks gradients: a stream fit varies M200, c and q0, and those
are exactly the parameters the spline path cannot be differentiated through.

Here the density itself is analytic, so the same fixed-node quadratures become
one traceable expression and the derivatives come out exactly.

The squashing is what makes this possible at all. The package defines the
squashed radius by a fixed point,

    r_sph = sqrt(R^2 q(r_sph)^(2/3) + z^2 q(r_sph)^(-4/3))

and iterates it. For a **constant** q the iteration is a no-op and the map is
closed form:

    r_sph(r, theta) = r sqrt(sin^2(theta) q^(2/3) + cos^2(theta) q^(-4/3))

which is differentiable in q0 directly. A radius-dependent q(r) would need the
fixed point unrolled to a fixed depth; not implemented, and the functions here
raise rather than pretend.

Scope: NFW, constant q0, no adiabatic contraction. Einasto and AC still need
the package path and so still block gradients in M200 and c -- AC in
particular shifts the dark-matter normalisation 7-12% at stream radii, so it
is not optional for real inference, only for getting the machinery working.
"""
import numpy as np

from .jaxsolver import GN, HAVE_JAX, N_GL_DEFAULT, _require_jax, gl_tables

if HAVE_JAX:
    import jax
    import jax.numpy as jnp

__all__ = ["nfw_params", "nfw_rho", "r_sph", "rho_sph_avg", "enclosed_mass",
           "potential_moments", "boundary_data"]

N_R_DEFAULT = 64


def nfw_params(M200, c, h=0.7, del_c=200.0, Om=0.3, Ol=0.7, z=0.0):
    """(rho_s, r_s), matching jeans.cdm.mass_concentration_to_NFW_parameters."""
    H0 = h * 100.0 * 1e-3
    rho_crit = (3.0 * H0 ** 2) / (8.0 * jnp.pi * GN) * (Om * (1.0 + z) ** 3 + Ol)
    over = (del_c / 3.0) * (c ** 3) / (jnp.log1p(c) - c / (1.0 + c))
    rvir = jnp.cbrt(3.0 * M200 / (del_c * 4.0 * jnp.pi * rho_crit))
    return rho_crit * over, rvir / c


def nfw_rho(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1.0 + x) ** 2)


def r_sph(r, theta, q0):
    """The package's squashed radius, closed form for constant q.

    r_sph = r sqrt(sin^2(theta) q^(2/3) + cos^2(theta) q^(-4/3))
    """
    st2 = jnp.sin(theta) ** 2
    ct2 = jnp.cos(theta) ** 2
    return r * jnp.sqrt(st2 * q0 ** (2.0 / 3.0) + ct2 * q0 ** (-4.0 / 3.0))


def _angular(n_gl):
    theta, w = gl_tables(n_gl)
    return jnp.asarray(theta), jnp.asarray(w)


def rho_sph_avg(r, M200, c, q0=1.0, n_gl=N_GL_DEFAULT):
    """< rho(r, theta) >_theta for a squashed NFW halo."""
    rho_s, r_s = nfw_params(M200, c)
    theta, w = _angular(n_gl)
    return jnp.sum(w * nfw_rho(r_sph(r, theta, q0), rho_s, r_s))


def enclosed_mass(r1, M200, c, q0=1.0, n_r=48, n_gl=16):
    """M(<r1) for a squashed NFW halo.

    Uses r = r1 v^2, which turns the r^-1 central cusp of
    4 pi r^2 rho dr into a smooth 8 pi r1^3 v^5 rho(r1 v^2) dv, so a fixed rule
    needs no inner cutoff and no separate tail integral. Same substitution as
    the numpy version in jeans.fast.outer, which agrees with the package to
    5.7e-11.
    """
    rho_s, r_s = nfw_params(M200, c)
    xv, wv_np = np.polynomial.legendre.leggauss(n_r)
    v = jnp.asarray(0.5 * (xv + 1.0))
    wv = jnp.asarray(0.5 * wv_np)
    theta, wx = _angular(n_gl)

    rk = r1 * v * v                                        # (n_r,)
    rr = r_sph(rk[:, None], theta[None, :], q0)            # (n_r, n_gl)
    rho_avg = jnp.sum(wx[None, :] * nfw_rho(rr, rho_s, r_s), axis=1)
    return 8.0 * jnp.pi * r1 ** 3 * jnp.sum(wv * v ** 5 * rho_avg)


def potential_moments(r1, M200, c, q0=1.0, L_list=(0,), n_r=N_R_DEFAULT,
                      n_gl=24):
    """J_L for each L, matching CDM_profile.compute_potential_moments.

    Uses t = r1/r, which maps the infinite tail onto t -> 0 where the integrand
    tends to t^L / r1^(L+1) for an r^-3 profile, so one fixed rule covers the
    whole range with no shell sequence.
    """
    from scipy.special import eval_legendre
    L_list = tuple(L_list)
    if any(L % 2 for L in L_list):
        raise ValueError("odd L are not supported (z-symmetry assumed)")
    rho_s, r_s = nfw_params(M200, c)

    xt, wt = np.polynomial.legendre.leggauss(n_r)
    t = jnp.asarray(0.5 * (xt + 1.0))
    wt = jnp.asarray(0.5 * wt)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    xg, wg = 0.5 * (xg + 1.0), 0.5 * wg
    theta = jnp.asarray(np.arccos(np.clip(xg, -1.0, 1.0)))
    wx = jnp.asarray(wg)
    Z = jnp.asarray([np.sqrt((2 * L + 1) / (4.0 * np.pi)) * eval_legendre(L, xg)
                     for L in L_list])

    r = r1 / t
    rr = r_sph(r[:, None], theta[None, :], q0)
    rho = nfw_rho(rr, rho_s, r_s)                           # (n_r, n_gl)

    out = []
    for k, L in enumerate(L_list):
        rho_L = 4.0 * jnp.pi * jnp.sum(wx[None, :] * Z[k][None, :] * rho, axis=1)
        out.append(r1 ** (2 - L) * jnp.sum(wt * t ** (L - 3) * rho_L))
    return jnp.stack(out)


def boundary_data(M200, c, r1, q0=1.0, L_list=(0,), n_r=N_R_DEFAULT,
                  n_gl=N_GL_DEFAULT):
    """(rho1, M1, J_L), all differentiable in M200, c, q0 and r1.

    For q0 == 1 this reproduces jaxsolver.nfw_boundary; for q0 != 1 it is the
    squashed halo the package builds, and it is the piece that was previously
    forcing a host call and cutting the gradient chain.
    """
    _require_jax()
    return (rho_sph_avg(r1, M200, c, q0, n_gl),
            enclosed_mass(r1, M200, c, q0),
            potential_moments(r1, M200, c, q0, L_list, n_r))
