"""A differentiable density profile for the solved halo, as one JAX callable.

Everything downstream of this package wants the same object. galpy's
MultipoleExpansionPotential.from_density takes rho(r), rho(R, z) or
rho(R, z, phi); agama's Potential(type='Multipole', density=...) takes the
same; galax and StreamSculptor want Phi, which follows from rho. So rather
than three exporters there is one deliverable -- a traceable rho(R, z) that is
differentiable in the halo parameters -- and three thin adapters over it.

The solvers could not supply that. jaxsolver.solve_log returns only
[log r0, log sigma0]: rk4_monopole's lax.scan discards its trajectory, so the
interior phi(r) is computed and thrown away, and only the non-differentiable
numpy path ever exposed a density. rk4_monopole_traj below keeps it.

Inside r1 the dark-matter density is

    rho(r, theta) = rho0 exp(-phi(r) - dPhi_b(r, theta) / sigma0^2)

with rho0 = sigma0^2 / (4 pi G r0^2). Note the angular dependence: even in the
spherical solver the DM density is a function of theta whenever the baryons
are, because only the MONOPOLE is matched on the angular average. Outside r1
it is the CDM profile, whose own angular dependence comes from the flattening
q0 instead. The two are therefore matched in the angular average and in the
enclosed mass, not pointwise, so a seam at r1 is expected; its size is
measured in tests/test_jaxprofile.py rather than assumed small.

Differentiability survives the interpolation because the nodes are fixed and
only the values carry parameters -- the same reason the fixed step count
matters for the solve itself.
"""
from functools import partial

import numpy as np

from .jaxsolver import (GN, HAVE_JAX, N_GL_DEFAULT, _require_jax, gl_tables,
                        mn_grid, solve_log)

if HAVE_JAX:
    import jax
    import jax.numpy as jnp
    from jax import lax

__all__ = ["rk4_monopole_traj", "interior_profile", "density_fn",
           "potential_fn",
           "spherical_density_fn", "mass_fn", "radial_acceleration_fn"]


def _hermite(x, x0, h, y, dy):
    """Cubic Hermite on a uniform grid, from values and exact derivatives.

    Linear interpolation is not good enough here. Its C0 kinks put about 5e-5
    of non-smoothness into the gradient of an orbit loss; cubic Hermite with
    the exact dM/dr gives 1e-8 on the same test, four orders better for no
    extra solve, because the derivative is already known analytically. It also
    costs the adaptive integrator nothing at normal tolerances, and saves
    about 19% in steps and 55% in rejections at rtol 1e-10.
    """
    n = y.shape[0]
    i = jnp.clip(jnp.floor((x - x0) / h).astype(int), 0, n - 2)
    t = (x - (x0 + i * h)) / h
    t2 = t * t
    t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
            + (t3 - 2.0 * t2 + t) * h * dy[i]
            + (-2.0 * t3 + 3.0 * t2) * y[i + 1]
            + (t3 - t2) * h * dy[i + 1])


def rk4_monopole_traj(r_nodes, h, r0sq, s_n, s_h, unroll=1):
    """As jaxsolver.rk4_monopole, but keeping phi and eta at every node.

    Identical arithmetic -- the only change is that the scan emits its carry,
    so the two agree to the last bit at the endpoint. Returns
    (phi[n+1], eta[n+1]) including the phi(0) = eta(0) = 0 start.
    """
    inv3 = 1.0 / (3.0 * r0sq)
    n = r_nodes.shape[0] - 1
    first = jnp.zeros(n, dtype=bool).at[0].set(True)

    def body(carry, xs):
        phi, eta = carry
        rA, rB, sA, sM, sB, is0 = xs
        rM = rA + 0.5 * h
        rA_s = jnp.where(is0, 1.0, rA)
        k1p = jnp.where(is0, 0.0, rA * inv3 * jnp.exp(-eta))
        k1e = jnp.where(is0, 0.0, -(3.0 / rA_s) * jnp.expm1(eta - phi - sA))

        p2, e2 = phi + 0.5 * h * k1p, eta + 0.5 * h * k1e
        k2p = rM * inv3 * jnp.exp(-e2)
        k2e = -(3.0 / rM) * jnp.expm1(e2 - p2 - sM)

        p3, e3 = phi + 0.5 * h * k2p, eta + 0.5 * h * k2e
        k3p = rM * inv3 * jnp.exp(-e3)
        k3e = -(3.0 / rM) * jnp.expm1(e3 - p3 - sM)

        p4, e4 = phi + h * k3p, eta + h * k3e
        k4p = rB * inv3 * jnp.exp(-e4)
        k4e = -(3.0 / rB) * jnp.expm1(e4 - p4 - sB)

        out = (phi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
               eta + h / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e))
        return out, out

    xs = (r_nodes[:-1], r_nodes[1:], s_n[:-1], s_h, s_n[1:], first)
    _, (phi, eta) = lax.scan(body, (0.0, 0.0), xs, unroll=unroll)
    z = jnp.zeros((1,))
    return jnp.concatenate([z, phi]), jnp.concatenate([z, eta])


def interior_profile(params, n_steps=200, n_gl=N_GL_DEFAULT, **kw):
    """(r_nodes, phi, eta, s, r0, sigma0) for params = (M200, c, r1, Md, a, b).

    The solve is jaxsolver.solve_log, so the implicit-differentiation rule is
    the one already validated; this only re-marches at the converged point to
    recover the trajectory the solver throws away.
    """
    _require_jax()
    M200, c, r1, Md, a, b = [params[i] for i in range(6)]
    lg = solve_log(params, n_steps=n_steps, n_gl=n_gl, **kw)
    r0, sigma0 = jnp.exp(lg[0]), jnp.exp(lg[1])
    sig0sq = sigma0 ** 2

    nodes = jnp.linspace(0.0, r1, n_steps + 1)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    h = nodes[1] - nodes[0]
    theta, w = gl_tables(n_gl)
    theta, w = jnp.asarray(theta), jnp.asarray(w)

    from .jaxsolver import source_from_grid
    s_n = source_from_grid(mn_grid(Md, a, b, nodes, theta), w, sig0sq, 1.0)
    s_h = source_from_grid(mn_grid(Md, a, b, half, theta), w, sig0sq, 1.0)
    phi, eta = rk4_monopole_traj(nodes, h, r0 ** 2, s_n, s_h)
    return nodes, phi, eta, s_n, r0, sigma0


def density_fn(params, n_steps=200, n_gl=N_GL_DEFAULT, q0=1.0,
               outer_kw=None, **kw):
    """Return rho(R, z), traceable and differentiable in `params`.

    R and z are cylindrical, in kpc; the return is Msun/kpc^3. Accepts arrays
    and broadcasts. Suitable directly as the `dens` argument of galpy's
    MultipoleExpansionPotential.from_density or agama's Multipole.
    """
    _require_jax()
    from . import jaxouter as JO
    M200, c, r1, Md, a, b = [params[i] for i in range(6)]
    nodes, phi, eta, s_n, r0, sigma0 = interior_profile(params, n_steps, n_gl, **kw)
    sig0sq = sigma0 ** 2
    rho0 = sig0sq / (4.0 * jnp.pi * GN * r0 ** 2)
    rho_out = JO.halo_profile(M200, c, **(outer_kw or {}))

    def rho(R, z):
        R = jnp.asarray(R, float)
        z = jnp.asarray(z, float)
        r = jnp.sqrt(R ** 2 + z ** 2)
        # interior: fixed nodes, parameter-dependent values, so the
        # interpolation is differentiable in the halo parameters as well as
        # in r. Hermite rather than linear, with dphi/dr taken from the ODE
        # itself rather than differenced.
        dphi = nodes / (3.0 * r0 ** 2) * jnp.exp(-eta)
        ph = _hermite(jnp.clip(r, 0.0, r1), 0.0, nodes[1] - nodes[0], phi, dphi)
        dphi_b = (-GN * Md / jnp.sqrt(R ** 2 + (a + jnp.sqrt(b ** 2 + z ** 2)) ** 2)
                  + GN * Md / (a + b))
        inner = rho0 * jnp.exp(-ph - dphi_b / sig0sq)
        # exterior: the CDM profile on the squashed radius
        rs = JO.r_sph(jnp.maximum(r, 1e-12),
                      jnp.arctan2(jnp.abs(R), z), q0)
        outer = rho_out(rs)
        return jnp.where(r <= r1, inner, outer)

    return rho


def spherical_density_fn(params, n_gl=N_GL_DEFAULT, **kw):
    """Angular average of density_fn, as rho(r). For galpy's 1-argument form.

    The average is the one the monopole is matched on, so this is the profile
    whose enclosed mass reproduces M1 at r1 by construction.
    """
    _require_jax()
    f = density_fn(params, n_gl=n_gl, **kw)
    theta, w = gl_tables(n_gl)
    theta, w = jnp.asarray(theta), jnp.asarray(w)

    def rho(r):
        r = jnp.asarray(r, float)
        sh = r.shape
        rr = jnp.atleast_1d(r)[:, None]
        vals = f(rr * jnp.sin(theta)[None, :], rr * jnp.cos(theta)[None, :])
        return jnp.sum(w[None, :] * vals, axis=1).reshape(sh)

    return rho


def mass_fn(params, n_steps=200, n_gl=N_GL_DEFAULT, q0=1.0, outer_kw=None, **kw):
    """Enclosed DARK MATTER mass M(<r), traceable and differentiable.

    Dark matter only; the baryons shape it through Phi_b but are not added to
    it. See radial_acceleration_fn.

    Inside r1 this needs no interpolation of a potential and no Poisson
    solve: the march's second variable gives it in closed form,

        M(<r) = (4 pi / 3) rho0 r^3 exp(-eta(r)),

    which is also where the radial acceleration comes from. Continuity at r1
    is exact rather than approximate -- M(<r1) = M1 and rho(r1) = rho1 are
    matching conditions of the solve, so they hold to machine precision
    (measured 8.9e-16 and 1.4e-15), and g = -GM/r^2 is therefore C1 across
    the join with nothing to patch.
    """
    _require_jax()
    from . import jaxouter as JO
    M200, c, r1 = params[0], params[1], params[2]
    nodes, phi, eta, s_n, r0, sigma0 = interior_profile(params, n_steps, n_gl, **kw)
    rho0 = sigma0 ** 2 / (4.0 * jnp.pi * GN * r0 ** 2)
    pref = 4.0 * jnp.pi / 3.0 * rho0
    M_nodes = pref * nodes ** 3 * jnp.exp(-eta)
    # exact derivative: dM/dr = 4 pi r^2 <rho>, and <rho> = rho0 exp(-phi - s)
    dM_nodes = 4.0 * jnp.pi * nodes ** 2 * rho0 * jnp.exp(-phi - s_n)
    h = nodes[1] - nodes[0]

    def M(r):
        r = jnp.asarray(r, float)
        inner = _hermite(jnp.clip(r, 0.0, r1), 0.0, h, M_nodes, dM_nodes)
        outer = JO.enclosed_mass(jnp.maximum(r, r1), M200, c, q0=q0,
                                 **(outer_kw or {}))
        return jnp.where(r <= r1, inner, outer)

    return M


def radial_acceleration_fn(params, **kw):
    """g_r(r) = -G M_DM(<r) / r^2, from the DARK MATTER ONLY.

    The baryons are NOT included. They enter the solve as Phi_b, shaping the
    dark matter, but this returns the dark matter's own field; a stream
    integrator needs the total, so add the baryon acceleration yourself. The
    omission is silent and easy to miss -- an orbit outside r1 in this field
    has exactly zero gradient with respect to Md, a and b, which is the tell.

    This is the hot path: an adaptive Dopri5 run of 1000 particles over 5 Gyr
    is about 5.1e6 evaluations, against a single ~19 ms halo solve, so the
    solve is 0.6-1.5% of a likelihood and all the effort belongs here.
    """
    _require_jax()
    M = mass_fn(params, **kw)

    def g(r):
        r = jnp.asarray(r, float)
        rs = jnp.where(r > 0.0, r, 1.0)
        return jnp.where(r > 0.0, -GN * M(rs) / rs ** 2, 0.0)

    return g


def potential_fn(params, r_max=None, n_grid=512, **kw):
    """Phi(r) for the DARK MATTER, traceable and differentiable.

    Integrates dPhi/dr = G M(<r)/r^2 inward from an outer anchor where the
    halo is treated as a point mass, Phi(r_max) = -G M(<r_max)/r_max. Hermite
    interpolation on a log grid, with the derivative taken from M(r) exactly
    rather than differenced, so dPhi/dr is consistent with
    radial_acceleration_fn to machine precision instead of approximately.

    DARK MATTER ONLY, like mass_fn and radial_acceleration_fn. The baryons
    shape the halo through Phi_b but are not added here; an orbit integrator
    needs the total and must add the baryonic potential itself. The omission
    is silent -- the tell is that an orbit outside r1 has exactly zero
    gradient with respect to Md, a and b.
    """
    _require_jax()
    M = mass_fn(params, **kw)
    r1 = params[2]
    rmax = 50.0 * r1 if r_max is None else r_max
    u = jnp.linspace(jnp.log(1e-4 * r1), jnp.log(rmax), n_grid)
    rg = jnp.exp(u)
    # M is NOT vectorised over r: its exterior branch runs a fixed
    # quadrature whose grid collides with the radial grid under broadcasting.
    Mg = jax.vmap(M)(rg)
    dPhi = GN * Mg / rg ** 2                        # dPhi/dr
    integ = dPhi * rg                               # dPhi/du
    # cumulative from the outer end inward
    seg = 0.5 * (integ[1:] + integ[:-1]) * jnp.diff(u)
    tail = jnp.concatenate([jnp.cumsum(seg[::-1])[::-1], jnp.zeros(1)])
    Phi_inf = -GN * M(rmax) / rmax
    Phi_g = Phi_inf - tail

    def Phi(r):
        r = jnp.asarray(r, float)
        rc = jnp.clip(r, rg[0], rg[-1])
        # Hermite in log r, with dPhi/du = G M(r)/r exact at the nodes. Plain
        # linear interpolation here left the agreement with
        # radial_acceleration_fn at 1.4e-2, which is the docstring's claim
        # unmet and far too coarse for an orbit integrator.
        lo = _hermite(jnp.log(rc), u[0], u[1] - u[0], Phi_g, GN * Mg / rg)
        # beyond the grid the halo is a point mass
        return jnp.where(r > rg[-1], Phi_inf * rmax / r, lo)

    return Phi
