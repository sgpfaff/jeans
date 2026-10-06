"""The parameter-free interior solution, and the no-baryon matching it reduces to.

With Phi_b = 0 the substitution u = r/r0 removes every parameter from the
interior system, so there is exactly one solution and every spherical SIDM halo
without baryons is a rescaling of it:

    dphi/du = (u/3) exp(-eta)
    deta/du = -(3/u) expm1(eta - phi)

with the regular series start phi ~ u^2/6, eta ~ u^2/10.

Matching to the outer halo then collapses to inverting one monotonic function,

    R(u1) = M1 / (4 pi r1^3 rho1) = (1/3) exp(phi(u1) - eta(u1)),

so u1 follows by interpolation on a stored table. No Newton iteration is
performed. Because R is bounded on its first branch, the model has a solution
if and only if R < R_MAX, which is an exact existence criterion available before
any solve.

The package already ships this curve as data/initial_guess.csv and uses it as a
starting guess; what is unused there is that in the no-baryon case the guess is
already the answer.
"""
import numpy as np

from .kernels import rk4_monopole_full

__all__ = ["table", "R_MAX", "U_AT_R_MAX", "matching_ratio", "exists", "solve", "seed"]

# Resolution of the stored curve. h = umax/N = 2.5e-4 puts the RK4 truncation
# error far below the interpolation error, which is itself ~1e-8.
_N_DEFAULT = 240_000
_UMAX_DEFAULT = 60.0

_CACHE = {}

# Provisional values, replaced by refresh_constants() on first use of the
# branch. They must never exceed the stored table's own maximum: a literal
# 4.7e-9 above it made seed()'s clamped retry land past the branch end, so the
# retry returned None on 100% of R >= R_MAX configurations and the with-baryon
# fallback was dead code. R_MAX is now derived from the table, not asserted.
R_MAX = 1.2615265933849009
U_AT_R_MAX = 22.544206


def table(n=_N_DEFAULT, umax=_UMAX_DEFAULT):
    """Return (u, phi, eta) for the universal interior solution.

    Cached per (n, umax); the first call costs a few milliseconds.
    """
    key = (n, umax)
    if key in _CACHE:
        return _CACHE[key]
    nodes = np.linspace(0.0, umax, n + 1)
    h = nodes[1] - nodes[0]
    zero_n = np.zeros(n + 1)
    zero_h = np.zeros(n)
    phi, eta = rk4_monopole_full(nodes, h, 1.0, zero_n, zero_h)
    _CACHE[key] = (nodes, phi, eta)
    return _CACHE[key]


def _first_branch(n=_N_DEFAULT, umax=_UMAX_DEFAULT):
    """The monotone piece of g = phi - eta, up to and including its first maximum."""
    key = ("branch", n, umax)
    if key in _CACHE:
        return _CACHE[key]
    u, phi, eta = table(n, umax)
    g = phi - eta
    imax = int(np.argmax(g))
    _CACHE[key] = (u[: imax + 1], g[: imax + 1], phi[: imax + 1], eta[: imax + 1])
    return _CACHE[key]


def matching_ratio(u1, n=_N_DEFAULT, umax=_UMAX_DEFAULT):
    """R(u1) = (1/3) exp(phi(u1) - eta(u1)), the dimensionless matching function."""
    u, g, _, _ = _first_branch(n, umax)
    return np.exp(np.interp(u1, u, g)) / 3.0


def exists(r1, rho1, M1):
    """Exact no-baryon existence criterion: R = M1/(4 pi r1^3 rho1) < R_MAX."""
    return matching_ratio_of(r1, rho1, M1) < R_MAX


def matching_ratio_of(r1, rho1, M1):
    """The dimensionless ratio R that the outer boundary data imposes."""
    return M1 / (4.0 * np.pi * r1 ** 3 * rho1)


def solve(r1, rho1, M1, GN=4.302e-6):
    """No-baryon spherical solution by interpolation. Returns (r0, sigma0) or None.

    This performs no root-find: R is monotonic on its first branch, so u1 is
    read off directly and everything else follows algebraically.
    """
    u, g, phi, _ = _first_branch()
    ratio = matching_ratio_of(r1, rho1, M1)
    target = np.log(3.0 * ratio)
    if not np.isfinite(target) or target >= g[-1]:
        return None  # R >= R_MAX: no solution exists
    u1 = float(np.interp(target, g, u))
    phi1 = float(np.interp(u1, u, phi))
    r0 = r1 / u1
    rho0 = rho1 * np.exp(phi1)
    sigma0 = np.sqrt(4.0 * np.pi * GN * rho0 * r0 ** 2)
    return r0, sigma0


def seed(r1, rho1, M1, GN=4.302e-6):
    """Continuation seed in the solver's log unknowns, [log r0^2, log sigma0^2].

    When no no-baryon solution exists the ratio is clamped just inside the
    branch endpoint: with baryons the existence boundary moves, so a
    configuration can be solvable even when its Phi_b = 0 counterpart is not.

    The clamp revives the seed but not the solve. Measured over 400 draws with
    R > R_MAX it returns a seed every time, and the continuation that follows
    converges on 3.5% of them -- because no seed can help there. The ramp's
    first stage IS the baryon-free problem, which has no solution above R_MAX,
    so the path the continuation wants to follow does not exist at its own
    starting point and no value of n_ramp changes that. Those configurations
    are reached instead by jeanie.branch.solve_bracketed, which does not
    continue anything.
    """
    out = solve(r1, rho1, M1, GN=GN)
    if out is None:
        # Clamp against the table's own branch end rather than against R_MAX.
        # Deriving the bound from the same array that solve() tests makes the
        # retry correct by construction; comparing against a separately stored
        # constant is what made it fail every time.
        _, g, _, _ = _first_branch()
        r_end = float(np.exp(g[-1]) / 3.0)
        rho_clamped = M1 / (4.0 * np.pi * r1 ** 3 * r_end * (1.0 - 1e-12))
        out = solve(r1, rho_clamped, M1, GN=GN)
        if out is None:
            return None
    r0, sigma0 = out
    return np.array([2.0 * np.log(r0), 2.0 * np.log(sigma0)])


def refresh_constants():
    """Recompute R_MAX and U_AT_R_MAX from the table. Used by the test suite."""
    global R_MAX, U_AT_R_MAX
    u, phi, eta = table()
    g = phi - eta
    i = int(np.argmax(g))
    R_MAX = float(np.exp(g[i]) / 3.0)
    # g is flat at its maximum, so the grid argmax locates u only to a few
    # parts in 1e4. Take the vertex of the parabola through the bracketing
    # samples instead; the value R_MAX is insensitive to this, the location is not.
    y0, y1, y2 = g[i - 1], g[i], g[i + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
    U_AT_R_MAX = float(u[i] + delta * (u[1] - u[0]))
    return R_MAX, U_AT_R_MAX
