"""Differentiable spherical solver: JAX backend.

Same algorithm as :mod:`jeans.fast.solver` -- fixed-node Gauss-Legendre angular
average, fixed-step RK4 monopole, two log-residuals in [log r0^2, log sigma0^2],
continuation in the baryon amplitude -- re-expressed so the whole forward solve
is one traceable expression. That buys three things the numba path cannot
offer: exact gradients by implicit differentiation, ``vmap`` over an ensemble,
and ``jit``.

Measured against numba at matched settings, pinned to one core, five runs:

    forward solve          23.7 +/- 0.4 ms   against numba's 30.8 +/- 0.3
    full 2x6 Jacobian      26.3 +/- 0.7 ms   against ~368 ms of finite differences
    vmap over 64            5.4 +/- 0.1 ms per halo
    agreement on r0        2.5e-12

The Jacobian is the point. Twelve central-difference solves cost 368 ms and
lose digits at small steps; implicit differentiation costs 1.15x one forward
solve and is exact.

What had to change, and why
---------------------------
Fixed work everywhere: no convergence test, no line search, no adaptive
quadrature. The numba solver's damped Newton with backtracking becomes
``n_newton`` undamped steps per continuation stage.

A measured correction to the usual justification for that, because this
module's own design notes had it wrong. What buys the smooth finite-difference
plateau is the fixed *discretisation* -- step count and quadrature nodes -- not
the fixed *iteration count*. A residual-tolerance stopping test was built and
scanned over 257 points: no cliffs at all, down to s = 1e-7. Newton converges
quadratically along a path that is itself smooth in the parameters, so stopping
part way along it is smooth too. Changing the step count from 200 to 400, by
contrast, moves log r0 by 1.09e-7, which is 6.8x the entire s=1e-7 signal.
Fixed iteration counts are still worth having, but for a different reason: a
vmapped ``while_loop`` costs 2.9x because every lane waits for the slowest.

Three failure modes this module exists to not have
--------------------------------------------------
Each was found by an adversarial verifier on a prototype that had it.

1. A masked failure must poison its own gradient. ``jnp.where(bad, nan, out)``
   returns NaN values but *zero* tangents, because ``where`` treats the NaN
   branch as a constant. A sampler then reads "insensitive to all six
   parameters" rather than "this halo failed", which is the one error mode that
   silently corrupts a posterior. See :func:`_poison`.

2. ``lax.custom_root`` never checks that it found a root. At a non-converged
   point it returns a clean-looking Jacobian that was measured wrong by factors
   of 10 to 7000. The implicit function theorem holds *at a root*; this module
   masks by default rather than assuming one was reached.

3. Past a saddle-node fold the residual stops being a correctness test.
   Spurious roots satisfy it to 1e-15 while being wrong by factors of 50 to
   2000, and every other cheap diagnostic looks clean. What separates them is
   that they are artefacts of the continuation path, so they move when the
   schedule changes. See :func:`solve_verified`.
"""
from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.special import logsumexp
    jax.config.update("jax_enable_x64", True)
    HAVE_JAX = True
except ImportError:                                   # pragma: no cover
    HAVE_JAX = False
    jax = jnp = lax = None

__all__ = ["HAVE_JAX", "solve_log", "solve_raw", "solve", "solve_verified",
           "nfw_boundary", "PARAM_NAMES"]

GN = 4.302e-6
N_GL_DEFAULT = 16
UNIV_STRIDE = 8          # subsample of the universal curve, seed only
RESID_TOL = 1e-8

PARAM_NAMES = ("M200", "c", "r1", "Md", "a", "b")

_GL_CACHE = {}
_UNIV_CACHE = {}


def _require_jax():
    if not HAVE_JAX:                                  # pragma: no cover
        raise ImportError(
            "the JAX backend needs jax installed; the numba backend in "
            "jeans.fast.solver has no such dependency"
        )


# --------------------------------------------------------------- constants
def gl_tables(n=N_GL_DEFAULT):
    """Angular nodes and weights, identical to jeans.fast.quadrature."""
    if n not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        x, w = 0.5 * (x + 1.0), 0.5 * w
        _GL_CACHE[n] = (np.ascontiguousarray(np.arccos(np.clip(x, -1.0, 1.0))),
                        np.ascontiguousarray(w))
    return _GL_CACHE[n]


def universal_branch(stride=UNIV_STRIDE):
    """(u, g = phi - eta, phi) on the monotone first branch.

    Taken from jeans.fast.universal so both backends seed from the same curve.
    Parameter-free, so it is a compile-time constant. The branch endpoint is
    always kept whatever the stride, because it is the existence criterion.
    """
    if stride not in _UNIV_CACHE:
        from jeans.fast import universal as _u
        u, g, phi, _ = _u._first_branch()
        sl = slice(None, None, stride)
        uu, gg, pp = u[sl].copy(), g[sl].copy(), phi[sl].copy()
        if uu[-1] != u[-1]:
            uu = np.append(uu, u[-1])
            gg = np.append(gg, g[-1])
            pp = np.append(pp, phi[-1])
        _UNIV_CACHE[stride] = tuple(np.ascontiguousarray(v) for v in (uu, gg, pp))
    return _UNIV_CACHE[stride]


# --------------------------------------------------------------- outer halo
def nfw_boundary(M200, c, r1, h=0.7, del_c=200.0, Om=0.3, Ol=0.7, z=0.0):
    """(rho1, M1) for an NFW outer halo, in closed form.

    Differentiable in M200 and c, which the package's spline-based path is not
    -- and those are exactly the parameters an inference run varies.
    Agrees with CDM_profile.rho_sph_avg to 0 and .M_encl to ~6e-11.
    """
    H0 = h * 100.0 * 1e-3
    rho_crit = (3.0 * H0 ** 2) / (8.0 * jnp.pi * GN) * (Om * (1.0 + z) ** 3 + Ol)
    over = (del_c / 3.0) * (c ** 3) / (jnp.log1p(c) - c / (1.0 + c))
    rvir = jnp.cbrt(3.0 * M200 / (del_c * 4.0 * jnp.pi * rho_crit))
    rs = rvir / c
    rhos = rho_crit * over
    x = r1 / rs
    return rhos / x / (1.0 + x) ** 2, \
        4.0 * jnp.pi * rhos * rs ** 3 * (jnp.log1p(x) - x / (1.0 + x))


# ------------------------------------------------------------ baryon sector
def mn_grid(Md, a, b, r, theta):
    """dPhi_b(r, theta) = Phi_MN(r, theta) - Phi_MN(0) on an (n_r, n_th) grid."""
    R = r[:, None]
    st, ct = jnp.sin(theta)[None, :], jnp.cos(theta)[None, :]
    denom = jnp.sqrt(R ** 2 * st ** 2 + (a + jnp.sqrt(b ** 2 + R ** 2 * ct ** 2)) ** 2)
    return -GN * Md / denom + GN * Md / (a + b)


def source_from_grid(grid, w, sig0sq, upsilon):
    """s(r) = -log < exp(-Upsilon dPhi_b / sigma0^2) >_theta, stably."""
    return -logsumexp(-upsilon * grid / sig0sq, b=w[None, :], axis=1)


# -------------------------------------------------------------------- RK4
def rk4_monopole(r_nodes, h, r0sq, s_n, s_h, unroll=1):
    """Fixed-step RK4 for the monopole. Returns (phi1, eta1).

    The r=0 step uses the double-where idiom: a safe denominator inside and a
    select outside, so the 0/0 produces neither a NaN value nor a NaN tangent.
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

        return (phi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
                eta + h / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)), None

    xs = (r_nodes[:-1], r_nodes[1:], s_n[:-1], s_h, s_n[1:], first)
    (phi1, eta1), _ = lax.scan(body, (0.0, 0.0), xs, unroll=unroll)
    return phi1, eta1


# --------------------------------------------------------------- residual
def _make_residual(r_nodes, h, grid_n, grid_h, w, r1, rho1, M1, unroll=1):
    log_rho1, log_M1, log_r1 = jnp.log(rho1), jnp.log(M1), jnp.log(r1)

    def F(logp, upsilon):
        r0sq, sig0sq = jnp.exp(logp[0]), jnp.exp(logp[1])
        s_n = source_from_grid(grid_n, w, sig0sq, upsilon)
        s_h = source_from_grid(grid_h, w, sig0sq, upsilon)
        phi1, eta1 = rk4_monopole(r_nodes, h, r0sq, s_n, s_h, unroll=unroll)
        log_rho0 = jnp.log(sig0sq / (4.0 * jnp.pi * GN * r0sq))
        return jnp.stack([
            log_rho0 - phi1 - s_n[-1] - log_rho1,
            jnp.log(4.0 * jnp.pi / 3.0) + log_rho0 + 3.0 * log_r1 - eta1 - log_M1,
        ])

    return F


def _value_and_jac(fun, x):
    """F(x) and the exact 2x2 dF/dx as one vmapped JVP.

    Forward mode through lax.scan is a single scan with an augmented carry and
    no residual storage, so the 200-step loop runs once. jax.linearize instead
    splits it into a primal scan that stores 200 residual tuples and a second
    linear scan that reads them back, which measured 2.6x slower.
    """
    Y, dY = jax.vmap(lambda t: jax.jvp(fun, (x,), (t,)))(jnp.eye(2))
    return Y[0], dY.T


# ------------------------------------------------------------------- seed
def universal_seed(r1, rho1, M1, tables):
    """[log r0^2, log sigma0^2] from the universal curve, branch-free.

    Mirrors jeans.fast.universal.seed including its clamp: when no Phi_b = 0
    solution exists the target is pinned just inside the branch endpoint,
    because baryons move the existence boundary. The clamp is taken against
    g[-1] itself rather than against a stored R_MAX -- a constant that sat
    4.7e-9 above the table made the equivalent numba retry fail on 988 of 988
    draws.
    """
    u, g, phi = tables
    ratio = M1 / (4.0 * jnp.pi * r1 ** 3 * rho1)
    target = jnp.minimum(jnp.log(3.0 * ratio), float(g[-1]) + np.log1p(-1e-9))
    u1 = jnp.interp(target, g, u)
    phi1 = jnp.interp(u1, u, phi)
    r0 = r1 / u1
    sig0sq = 4.0 * jnp.pi * GN * rho1 * jnp.exp(phi1) * r0 ** 2
    return jnp.stack([2.0 * jnp.log(r0), jnp.log(sig0sq)]), ratio


# ------------------------------------------------------------- the solver
def _solve_core(params, n_steps, n_gl, n_ramp, n_newton, unroll, implicit):
    M200, c, r1, Md, a, b = params
    theta, w = gl_tables(n_gl)
    tables = universal_branch()

    rho1, M1 = nfw_boundary(M200, c, r1)
    r_nodes = jnp.linspace(0.0, r1, n_steps + 1)
    r_half = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    h = r1 / n_steps

    F = _make_residual(r_nodes, h,
                       mn_grid(Md, a, b, r_nodes, theta),
                       mn_grid(Md, a, b, r_half, theta),
                       w, r1, rho1, M1, unroll=unroll)
    x0, ratio = universal_seed(r1, rho1, M1, tables)

    def newton(x, upsilon):
        def step(xk, _):
            R, J = _value_and_jac(lambda z: F(z, upsilon), xk)
            return xk - jnp.linalg.solve(J, R), None
        x, _ = lax.scan(step, x, None, length=n_newton)
        return x

    def ramp(x_init):
        ups = jnp.arange(1, n_ramp + 1, dtype=jnp.float64) / n_ramp

        def stage(carry, upsilon):
            x, prev, first = carry
            start = jnp.where(first, x, x + (x - prev))     # secant predictor
            return (newton(start, upsilon), x, jnp.array(False)), None

        (xf, _, _), _ = lax.scan(stage, (x_init, x_init, jnp.array(True)), ups)
        return xf

    F1 = lambda z: F(z, 1.0)
    if implicit:
        # custom_root attaches the implicit-function-theorem rule, so the
        # Newton iterations are never differentiated through.
        xf = lax.custom_root(
            F1, x0, lambda f, g0: ramp(g0),
            lambda g, y: jnp.linalg.solve(jax.jacobian(g)(y), y))
    else:
        xf = ramp(x0)

    return xf, jnp.max(jnp.abs(F1(xf))), ratio


def _poison(out, bad):
    """Mask so that BOTH the value and its tangent become NaN.

    jnp.where(bad, nan, out) gives NaN values and ZERO tangents, because the
    NaN branch is a constant with no derivative. Downstream that reads as "this
    halo is insensitive to every parameter" instead of "this halo failed", and
    a sampler or Fisher matrix will happily consume it. Multiplying instead
    puts the NaN on the path the tangent travels: d(out * nan)/dp = nan.
    """
    return out * jnp.where(bad, jnp.nan, 1.0)


@partial(jax.jit, static_argnums=tuple(range(1, 7))) if HAVE_JAX else (lambda f: f)
def solve_raw(params, n_steps=200, n_gl=N_GL_DEFAULT, n_ramp=16, n_newton=3,
              unroll=1, implicit=True):
    """([log r0, log sigma0], residual, ratio). Unmasked, fixed-work output."""
    xf, resid, ratio = _solve_core(params, n_steps, n_gl, n_ramp, n_newton,
                                   unroll, implicit)
    return jnp.stack([0.5 * xf[0], 0.5 * xf[1]]), resid, ratio


@partial(jax.jit, static_argnums=tuple(range(1, 8))) if HAVE_JAX else (lambda f: f)
def solve_log(params, n_steps=200, n_gl=N_GL_DEFAULT, n_ramp=16, n_newton=3,
              unroll=1, implicit=True, mask=True):
    """[log r0, log sigma0] for params = (M200, c, r1, Md, a, b).

    This is the differentiable entry point. Use it, not a convenience wrapper:
    a prototype whose dict-returning ``solve`` bypassed its own implicit rule
    differentiated through all 48 Newton steps instead, a silent 4.4x for the
    same answer.

    ``mask=True`` (the default) replaces a non-converged answer with NaN in
    both the value and the gradient. It is a data-dependent *output*, not
    control flow, so jit and vmap are unaffected; it does put a cliff in the
    derivative at the existence boundary, which is correct -- the solution
    genuinely ceases to exist there.
    """
    xf, resid, _ = _solve_core(params, n_steps, n_gl, n_ramp, n_newton,
                               unroll, implicit)
    out = jnp.stack([0.5 * xf[0], 0.5 * xf[1]])
    if mask:
        out = _poison(out, (resid > RESID_TOL) | ~jnp.isfinite(resid))
    return out


def solve(params, **kw):
    """Convenience scalar wrapper: dict of r0, sigma0, residual, ratio, ok.

    For gradients use :func:`solve_log` directly.
    """
    _require_jax()
    lg, resid, ratio = solve_raw(jnp.asarray(params, dtype=jnp.float64), **kw)
    resid = float(resid)
    return {"r0": float(jnp.exp(lg[0])), "sigma0": float(jnp.exp(lg[1])),
            "residual": resid, "ratio": float(ratio),
            "ok": bool(np.isfinite(resid) and resid <= RESID_TOL)}


def solve_verified(params, n_ramp=16, rtol=1e-6, **kw):
    """Solve twice at different ramp densities and require agreement.

    Past a fold the residual is not a correctness test: spurious roots satisfy
    it to 1e-15 while being wrong by factors of 50 to 2000, and they pass every
    other cheap diagnostic. They are artefacts of the continuation path, so
    they move when the schedule changes; a genuine root does not.

    A necessary condition, not a sufficient one -- two schedules can land on
    the same wrong branch -- so this is a filter, not a proof. Costs one extra
    solve. The equivalent gate in the numba backend caught 1/1 of the spurious
    successes in a 120-draw scan at a 0.7% false-positive rate.
    """
    _require_jax()
    lo = solve(params, n_ramp=n_ramp, **kw)
    hi = solve(params, n_ramp=2 * n_ramp, **kw)
    agree = (lo["ok"] and hi["ok"]
             and abs(hi["r0"] / lo["r0"] - 1.0) <= rtol)
    out = dict(lo)
    out["ok"] = bool(agree)
    out["r0_dense"] = hi["r0"]
    if lo["ok"] and not agree:
        out["reason"] = ("schedule_dependent_root(r0=%.6g at n_ramp=%d)"
                         % (hi["r0"], 2 * n_ramp))
    return out
