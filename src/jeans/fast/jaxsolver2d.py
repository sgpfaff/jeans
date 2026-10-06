"""Differentiable axisymmetric solver: JAX backend.

The two-dimensional counterpart of :mod:`jeans.fast.jaxsolver`, and the one
that matters scientifically -- the quadrupole is what a stream's orbital plane
responds to.

The structure turns out to be friendlier than it looks. Although the package
relaxes all 2(Lmax/2 + 1) multipole fields at once, the reduced formulation has
only **two** root-find unknowns, the same as in 1D: [log r0^2, log sigma0^2].
The shape is not an unknown at all. Linearised about the spherical background
each L>0 mode is a two-point linear problem, so it is solved in closed form as
one particular march plus one homogeneous march combined to meet the outer
boundary condition. What couples the sectors is that the monopole's angular
average feels the shape, which is handled by a fixed number of inner passes
inside a single residual evaluation rather than by an outer loop.

That means ``lax.custom_root`` applies unchanged: it still wraps a 2-unknown
root-find, and the shape solve is simply part of evaluating the residual.

The system, in the package's own variables so results compare directly:

    dphi_L/dr = mu_L / r^2
    dmu_L/dr  = L(L+1) phi_L - W_L(r) phi_L r^2 + r^2 S_L(r)
    S_L = (4 pi / r0^2) <Z_L exp(-phi_b - phi_00 Z_0)>
    W_L = (4 pi / r0^2) <Z_L^2 exp(-phi_b - phi_00 Z_0)>

with phi_L(0) = 0 and, at the matching radius,
mu_L(r1) + r1 (L+1) phi_L(r1) + 4 pi G r1^(L+1) J_L / sigma0^2 = 0.

Three things are load-bearing
-----------------------------
**The L>0 angular source must be formed as a difference.** The quadrature
satisfies sum_j w_j Z_L(x_j) = 0 identically for L>0, so evaluating
<Z_L exp(-v)> directly is pure round-off wherever the anisotropy is small --
which is the entire inner region. Computed naively the shape never moves.
Subtracting the angular mean first is what makes the sector solvable at all.

**The monopole must feel the shape.** Solving the two sectors once each leaves
r0 wrong by 2.9e-3, an order of magnitude above the package's own error and
matching the +3.6e-3 shift the package itself shows between L=[0] and L=[0,2].

**The homogeneous denominator can vanish.** ``mh + r1 (L+1) ph`` is divided
into, and at a degenerate configuration it approaches zero, so it gets the same
double-where treatment as the r=0 node.

Scope: even L only (z-symmetry assumed), NFW outer halo, Miyamoto-Nagai disc.
L=4 inherits the linear-response limitation documented in
:mod:`jeans.fast.solver2d` -- where phi_2^2 is comparable to phi_4 the dropped
quadratic term is the leading contribution to L=4, not a correction to it.
"""
from functools import partial

import numpy as np

from .jaxsolver import (GN, HAVE_JAX, N_GL_DEFAULT, RESID_TOL, _poison,
                        _require_jax, _value_and_jac, gl_tables, mn_grid,
                        nfw_boundary, rk4_monopole, source_from_grid,
                        universal_branch, universal_seed)

if HAVE_JAX:
    import jax
    import jax.numpy as jnp
    from jax import lax

__all__ = ["solve_log_2d", "solve_2d", "solve_2d_verified"]

_Z00 = 1.0 / np.sqrt(4.0 * np.pi)


def _harmonics(L_list, n_gl):
    """Z_L(x_j) on the angular nodes, M = 0. Compile-time constant."""
    from scipy.special import eval_legendre
    theta, _ = gl_tables(n_gl)
    x = np.cos(theta)
    return np.ascontiguousarray(
        [np.sqrt((2 * L + 1) / (4.0 * np.pi)) * eval_legendre(L, x) for L in L_list])


def _angular_terms(grid, w, Z, phi00, sig0sq, r0sq):
    """(S, W), each (n_L, n_r). See the module docstring on the L>0 difference.

    S_L for L>0 is formed against the angular mean because sum_j w_j Z_L = 0
    identically, so the direct sum is cancellation noise wherever the
    anisotropy is small.
    """
    pref = 4.0 * jnp.pi / r0sq
    v = grid / sig0sq
    vref = jnp.min(v, axis=1, keepdims=True)
    e = jnp.exp(-(v - vref))                      # O(1), cannot overflow
    mono = jnp.exp(-vref[:, 0] - phi00 * _Z00)
    ebar = jnp.sum(w[None, :] * e, axis=1)

    def one(k_is_monopole, ZL):
        direct = jnp.sum(w[None, :] * ZL[None, :] * e, axis=1)
        diff = jnp.sum(w[None, :] * ZL[None, :] * (e - ebar[:, None]), axis=1)
        s = jnp.where(k_is_monopole, direct, diff)
        ww = jnp.sum(w[None, :] * ZL[None, :] ** 2 * e, axis=1)
        return pref * mono * s, pref * mono * ww

    flags = jnp.arange(Z.shape[0]) == 0
    S, W = jax.vmap(one)(flags, Z)
    return S, W


def _march_mode(r_nodes, h, L, W_n, W_h, src_n, src_h):
    """One L>0 mode: particular and homogeneous solutions over the whole grid.

    Returns (phi_p, mu_p_end, phi_h, mu_h_end) with the trajectories as arrays
    and the endpoint momenta as scalars, so the outer boundary condition can be
    imposed afterwards by a linear combination.

    The homogeneous solution starts from the regular series r^L at the first
    non-zero node rather than being integrated through the origin, so the r^L
    scaling is exact.
    """
    LL = float(L * (L + 1))
    n = r_nodes.shape[0] - 1
    r_start = r_nodes[1]
    first = jnp.zeros(n, dtype=bool).at[0].set(True)

    def body(carry, xs):
        pp, mp, ph, mh = carry
        rA, rB, Wa, Wb, Wm, sa, sb, sm, is0 = xs
        rM = rA + 0.5 * h
        rA_s = jnp.where(is0, 1.0, rA)

        def step(p, m, use_src):
            k1p = jnp.where(is0, 0.0, m / (rA_s * rA_s))
            k1m = jnp.where(is0, 0.0,
                            LL * p - Wa * p * rA * rA + use_src * sa)
            p2, m2 = p + 0.5 * h * k1p, m + 0.5 * h * k1m
            k2p = m2 / (rM * rM)
            k2m = LL * p2 - Wm * p2 * rM * rM + use_src * sm
            p3, m3 = p + 0.5 * h * k2p, m + 0.5 * h * k2m
            k3p = m3 / (rM * rM)
            k3m = LL * p3 - Wm * p3 * rM * rM + use_src * sm
            p4, m4 = p + h * k3p, m + h * k3m
            k4p = m4 / (rB * rB)
            k4m = LL * p4 - Wb * p4 * rB * rB + use_src * sb
            return (p + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p),
                    m + h / 6.0 * (k1m + 2 * k2m + 2 * k3m + k4m))

        pp_n, mp_n = step(pp, mp, 1.0)
        ph_n, mh_n = step(ph, mh, 0.0)
        # the homogeneous solution is seeded at node 1, so hold it over step 0
        ph_n = jnp.where(is0, r_start ** L, ph_n)
        mh_n = jnp.where(is0, float(L) * r_start ** (L + 1), mh_n)
        return (pp_n, mp_n, ph_n, mh_n), (pp_n, ph_n)

    xs = (r_nodes[:-1], r_nodes[1:], W_n[:-1], W_n[1:], W_h,
          src_n[:-1], src_n[1:], src_h, first)
    init = (0.0, 0.0, r_start ** L, float(L) * r_start ** (L + 1))
    (pp, mp, ph, mh), (traj_p, traj_h) = lax.scan(body, init, xs)
    zero = jnp.zeros(1)
    return (jnp.concatenate([zero, traj_p]), mp,
            jnp.concatenate([jnp.array([r_start ** L]), traj_h]), mh)


def _solve_core_2d(params, L_list, J_L, n_steps, n_gl, n_ramp, n_newton,
                   n_inner, implicit, boundary=None):
    """params = (M200, c, r1, Md, a, b). L_list and J_L are static/array."""
    M200, c, r1, Md, a, b = params
    theta, w_np = gl_tables(n_gl)
    w = jnp.asarray(w_np)
    Z = jnp.asarray(_harmonics(L_list, n_gl))
    tables = universal_branch()
    higher = [(k, L) for k, L in enumerate(L_list) if L > 0]

    # The closed-form boundary is a SPHERICAL NFW. A squashed outer halo
    # (q0 != 1) has different rho1 and M1, so it must be supplied; gradients
    # then do not flow to M200 and c through the squashing, which is a real
    # scope limit until the squashed boundary is also written in closed form.
    if boundary is None:
        rho1, M1 = nfw_boundary(M200, c, r1)
    else:
        rho1, M1 = boundary
    r_nodes = jnp.linspace(0.0, r1, n_steps + 1)
    r_half = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    h = r1 / n_steps

    grid_n = mn_grid(Md, a, b, r_nodes, theta)
    grid_h = mn_grid(Md, a, b, r_half, theta)
    log_rho1, log_M1, log_r1 = jnp.log(rho1), jnp.log(M1), jnp.log(r1)

    def shape_and_residual(logp, upsilon):
        """Residual of the two monopole conditions, with the shape solved inside.

        n_inner fixed passes of (monopole march -> linear shape solve ->
        updated angular average). Fixed work, so the whole thing stays
        traceable and the finite-difference plateau survives.
        """
        r0sq, sig0sq = jnp.exp(logp[0]), jnp.exp(logp[1])
        gn, gh = upsilon * grid_n, upsilon * grid_h

        extra_n = jnp.zeros_like(gn)
        extra_h = jnp.zeros_like(gh)
        phi_L = [jnp.zeros(n_steps + 1) for _ in L_list]
        phi1 = eta1 = s_last = 0.0

        for _ in range(n_inner):
            # monopole, with whatever shape the previous pass produced
            s_n = source_from_grid(gn + extra_n, w_np, sig0sq, 1.0)
            s_h = source_from_grid(gh + extra_h, w_np, sig0sq, 1.0)
            phi1, eta1 = rk4_monopole(r_nodes, h, r0sq, s_n, s_h)
            s_last = s_n[-1]

            if not higher:
                break

            # full monopole trajectory, in the package's multipole normalisation
            phi00 = _monopole_traj(r_nodes, h, r0sq, s_n, s_h) / _Z00
            phi00_h = jnp.interp(r_half, r_nodes, phi00)

            S_n, W_n = _angular_terms(gn, w, Z, phi00, sig0sq, r0sq)
            S_h, W_h = _angular_terms(gh, w, Z, phi00_h, sig0sq, r0sq)

            extra_n = jnp.zeros_like(gn)
            extra_h = jnp.zeros_like(gh)
            for k, L in higher:
                src_n = r_nodes ** 2 * S_n[k]
                src_h = r_half ** 2 * S_h[k]
                tp, mp, th_, mh = _march_mode(r_nodes, h, L, W_n[k], W_h[k],
                                              src_n, src_h)
                K = 4.0 * jnp.pi * GN * r1 ** (L + 1) * J_L[k] / sig0sq
                denom = mh + r1 * (L + 1) * th_[-1]
                # double-where: a degenerate configuration can drive denom to
                # zero, and a bare divide would give a NaN tangent as well as
                # a NaN value.
                safe = jnp.where(jnp.abs(denom) < 1e-300, 1.0, denom)
                cc = jnp.where(jnp.abs(denom) < 1e-300, 0.0,
                               -(mp + r1 * (L + 1) * tp[-1] + K) / safe)
                pl = tp + cc * th_
                phi_L[k] = pl
                extra_n = extra_n + sig0sq * jnp.outer(pl, Z[k])
                extra_h = extra_h + sig0sq * jnp.outer(
                    jnp.interp(r_half, r_nodes, pl), Z[k])

        log_rho0 = jnp.log(sig0sq / (4.0 * jnp.pi * GN * r0sq))
        R = jnp.stack([
            log_rho0 - phi1 - s_last - log_rho1,
            jnp.log(4.0 * jnp.pi / 3.0) + log_rho0 + 3.0 * log_r1 - eta1 - log_M1,
        ])
        return R, phi_L

    F = lambda logp, ups: shape_and_residual(logp, ups)[0]

    x0, ratio = universal_seed(r1, rho1, M1, tables)

    def newton(x, upsilon):
        def step(xk, _):
            R, Jm = _value_and_jac(lambda z: F(z, upsilon), xk)
            return xk - jnp.linalg.solve(Jm, R), None
        x, _ = lax.scan(step, x, None, length=n_newton)
        return x

    def ramp(x_init):
        ups = jnp.arange(1, n_ramp + 1, dtype=jnp.float64) / n_ramp

        def stage(carry, upsilon):
            x, prev, first = carry
            start = jnp.where(first, x, x + (x - prev))
            return (newton(start, upsilon), x, jnp.array(False)), None

        (xf, _, _), _ = lax.scan(stage, (x_init, x_init, jnp.array(True)), ups)
        return xf

    F1 = lambda z: F(z, 1.0)
    if implicit:
        xf = lax.custom_root(
            F1, x0, lambda f, g0: ramp(g0),
            lambda g, y: jnp.linalg.solve(jax.jacobian(g)(y), y))
    else:
        xf = ramp(x0)

    resid_vec, phi_L = shape_and_residual(xf, 1.0)
    return xf, jnp.max(jnp.abs(resid_vec)), ratio, phi_L


def _monopole_traj(r_nodes, h, r0sq, s_n, s_h):
    """phi(r) over the whole grid; the shape sector needs the trajectory."""
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
        phi = phi + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        eta = eta + h / 6.0 * (k1e + 2 * k2e + 2 * k3e + k4e)
        return (phi, eta), phi

    xs = (r_nodes[:-1], r_nodes[1:], s_n[:-1], s_h, s_n[1:], first)
    _, traj = lax.scan(body, (0.0, 0.0), xs)
    return jnp.concatenate([jnp.zeros(1), traj])


# ------------------------------------------------------------------- API
def solve_log_2d(params, L_list=(0, 2), J_L=None, n_steps=200,
                 n_gl=N_GL_DEFAULT, n_ramp=16, n_newton=3, n_inner=3,
                 implicit=True, mask=True, boundary=None):
    """[log r0, log sigma0] for the axisymmetric model. Differentiable.

    ``J_L`` is the outer halo's potential moments, one per L. None means a
    spherical outer halo, which drives no L>0 structure by itself -- with a
    disc present the disc drives it instead.

    ``boundary`` overrides (rho1, M1). The internal closed form is a spherical
    NFW, so a squashed outer halo needs its own boundary data passed in; see
    jeans.fast.outer.boundary_data. Gradients then stop at the boundary rather
    than reaching M200 and c.
    """
    _require_jax()
    L_list = tuple(L_list)
    if L_list[0] != 0:
        raise ValueError("L_list must start with 0")
    if any(L % 2 for L in L_list):
        raise ValueError("odd L are not supported (z-symmetry assumed)")
    JL = (jnp.zeros(len(L_list)) if J_L is None
          else jnp.asarray(J_L, dtype=jnp.float64))
    xf, resid, _, _ = _solve_core_2d(params, L_list, JL, n_steps, n_gl,
                                     n_ramp, n_newton, n_inner, implicit,
                                     boundary)
    out = jnp.stack([0.5 * xf[0], 0.5 * xf[1]])
    if mask:
        out = _poison(out, (resid > RESID_TOL) | ~jnp.isfinite(resid))
    return out


def solve_2d(params, L_list=(0, 2), J_L=None, boundary=None, **kw):
    """Scalar convenience wrapper: r0, sigma0, residual, phi_L, ok."""
    _require_jax()
    L_list = tuple(L_list)
    JL = (jnp.zeros(len(L_list)) if J_L is None
          else jnp.asarray(J_L, dtype=jnp.float64))
    xf, resid, ratio, phi_L = _solve_core_2d(
        jnp.asarray(params, dtype=jnp.float64), L_list, JL,
        kw.get("n_steps", 200), kw.get("n_gl", N_GL_DEFAULT),
        kw.get("n_ramp", 16), kw.get("n_newton", 3),
        kw.get("n_inner", 3), kw.get("implicit", True), boundary)
    resid = float(resid)
    return {"r0": float(jnp.exp(0.5 * xf[0])),
            "sigma0": float(jnp.exp(0.5 * xf[1])),
            "residual": resid, "ratio": float(ratio),
            "L_list": list(L_list),
            "phi_L": {L: np.asarray(phi_L[k]) for k, L in enumerate(L_list)},
            "r_nodes": np.linspace(0.0, float(params[2]), kw.get("n_steps", 200) + 1),
            "ok": bool(np.isfinite(resid) and resid <= RESID_TOL)}


def solve_2d_verified(params, L_list=(0, 2), J_L=None, boundary=None,
                      n_ramp=16, rtol=1e-6, **kw):
    """Solve at two ramp densities and require agreement. See jaxsolver."""
    lo = solve_2d(params, L_list, J_L, boundary, n_ramp=n_ramp, **kw)
    hi = solve_2d(params, L_list, J_L, boundary, n_ramp=2 * n_ramp, **kw)
    agree = lo["ok"] and hi["ok"] and abs(hi["r0"] / lo["r0"] - 1.0) <= rtol
    out = dict(lo)
    out["ok"] = bool(agree)
    out["r0_dense"] = hi["r0"]
    if lo["ok"] and not agree:
        out["reason"] = ("schedule_dependent_root(r0=%.6g at n_ramp=%d)"
                         % (hi["r0"], 2 * n_ramp))
    return out
