"""Reduced spherical solver: interpolation without baryons, continuation with them.

Without baryons the interior system is parameter-free, so the solve is an
interpolation on the universal curve and no iteration happens at all
(universal.solve).

With baryons the parameter-free property is gone but the scaling structure
survives: the family is exactly three-dimensional in
{Lambda = G Md / (sigma0^2 r0), alpha = a/r0, beta = b/r0}. The unknowns are
still just (r0, sigma0), so the 402-dimensional Newton relaxation the package
performs reduces to a two-parameter root-find on fixed RK4 nodes.

Two details are load-bearing.

Continuation, not expansion. A Milky-Way-like disc sits at Lambda = 5.56, where
a first-order expansion in baryon depth is 101% wrong and second order is 17%
wrong; reaching 1e-3 needs Lambda below about 0.22. So the baryon sector is
ramped in rather than expanded. The ramp is also what selects the physical
branch: beyond a fold there are genuinely multiple roots, all with residuals
around 1e-15, and a direct Newton from a no-baryon seed picks a wrong one about
0.7% of the time while reporting success. A 16-stage ramp gave 0 spurious roots
in 314/314 cases and agreed with the package on 336/336 success-or-failure
outcomes.

Fixed step count. The node count never changes with the parameters, which is
what keeps finite differences of the output smooth down to s = 1e-7 where the
package's own derivatives flip sign.
"""
from dataclasses import dataclass, field

import numpy as np

from . import branch, universal
from .kernels import rk4_monopole, rk4_monopole_full, source_from_grid
from .quadrature import N_GL_DEFAULT, tabulate_baryons

__all__ = ["SphericalResult", "solve_spherical"]

GN = 4.302e-6  # km^2/s^2 kpc / Msun, matching jeans.definitions


@dataclass
class SphericalResult:
    r0: float
    sigma0: float
    success: bool
    reason: str = "ok"
    residual: float = np.nan
    ratio: float = np.nan
    n_residual_evals: int = 0
    rho0: float = np.nan
    r_nodes: np.ndarray = field(default=None, repr=False)
    phi: np.ndarray = field(default=None, repr=False)
    eta: np.ndarray = field(default=None, repr=False)

    def rho(self, r):
        """Dark-matter density at radius r, for r <= r1."""
        if self.phi is None:
            raise RuntimeError("solve with trajectory=True to evaluate rho(r)")
        phi = np.interp(r, self.r_nodes, self.phi)
        return self.rho0 * np.exp(-phi)


class _Problem:
    """Fixed per-solve data: nodes, outer boundary values, tabulated baryon grids."""

    __slots__ = ("r1", "rho1", "M1", "nodes", "half", "h",
                 "grid_n", "grid_h", "w", "upsilon", "n_eval",
                 "_isotropic", "_col_n", "_col_h", "_cache", "extra_n", "extra_h")

    def __init__(self, r1, rho1, M1, n_steps, Phi_b, n_gl, half_range):
        self.r1 = float(r1)
        self.rho1 = float(rho1)
        self.M1 = float(M1)
        self.nodes = np.linspace(0.0, self.r1, n_steps + 1)
        self.half = 0.5 * (self.nodes[:-1] + self.nodes[1:])
        self.h = self.nodes[1] - self.nodes[0]
        self.upsilon = 1.0
        self.n_eval = 0
        self._cache = {}
        # Shape feedback on the monopole. The axisymmetric solver puts
        # sigma0^2 * sum_{L>0} phi_L(r) Z_L(theta) here, which turns the
        # monopole's angular average into the effective one,
        # m0 = <exp(-phi_b - sum_{L>0} phi_L Z_L)>. Without it the monopole
        # never feels the shape and r0 is wrong by ~3.4e-3 for a disc.
        self.extra_n = None
        self.extra_h = None
        if Phi_b is None:
            self.grid_n = np.zeros((n_steps + 1, 1))
            self.grid_h = np.zeros((n_steps, 1))
            self.w = np.ones(1)
        else:
            self.grid_n, self.grid_h, self.w = tabulate_baryons(
                Phi_b, self.nodes, self.half, n=n_gl, half_range=half_range
            )
        # A theta-independent Phi_b makes the angular average exact and trivial:
        # <exp(-dPhi_b/sig0^2)> = exp(-dPhi_b/sig0^2), so s(r) = dPhi_b(r)/sig0^2
        # with no exponentials at all. Worth detecting, since the log-sum-exp is
        # two thirds of the residual cost and every spheroidal host hits this path.
        self._isotropic = bool(
            self.grid_n.shape[1] == 1
            or (np.ptp(self.grid_n, axis=1).max() == 0.0
                and np.ptp(self.grid_h, axis=1).max() == 0.0)
        )
        self._col_n = np.ascontiguousarray(self.grid_n[:, 0])
        self._col_h = np.ascontiguousarray(self.grid_h[:, 0])

    def sources(self, sig0sq):
        """Baryon source s(r) = -log m0(r) at nodes and half-nodes.

        Cached on sigma0^2: the Jacobian perturbs log r0^2 with sigma0 held
        fixed, so half of every finite-difference Jacobian reuses these arrays
        unchanged. The cache is cleared whenever the continuation level moves.
        """
        hit = self._cache.get(sig0sq)
        if hit is not None:
            return hit
        if self.extra_n is not None:
            u = self.upsilon
            gn = (self.grid_n if u == 1.0 else u * self.grid_n) + self.extra_n
            gh = (self.grid_h if u == 1.0 else u * self.grid_h) + self.extra_h
            out = (source_from_grid(gn, self.w, sig0sq),
                   source_from_grid(gh, self.w, sig0sq))
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[sig0sq] = out
            return out
        if self._isotropic:
            inv = self.upsilon / sig0sq
            out = (self._col_n * inv, self._col_h * inv)
        else:
            u = self.upsilon
            gn = self.grid_n if u == 1.0 else u * self.grid_n
            gh = self.grid_h if u == 1.0 else u * self.grid_h
            out = (source_from_grid(gn, self.w, sig0sq),
                   source_from_grid(gh, self.w, sig0sq))
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[sig0sq] = out
        return out

    def set_upsilon(self, value):
        self.upsilon = value
        self._cache.clear()

    def set_shape_feedback(self, extra_n, extra_h):
        """Install sigma0^2 * sum_{L>0} phi_L Z_L on the angular grid."""
        self.extra_n = extra_n
        self.extra_h = extra_h
        self._cache.clear()


def _residual(logp, P):
    """Two log-residuals for logp = [log r0^2, log sigma0^2].

    Posed in logs so that both equations are scale-free and the Jacobian is
    well conditioned: measured cond(J) over 749 converged prior draws has
    median 9.1 and p99 49.3, diverging only within ~1% of a fold.
    """
    P.n_eval += 1
    lp0 = float(np.clip(logp[0], -60.0, 60.0))
    lp1 = float(np.clip(logp[1], -60.0, 60.0))
    # Penalty keeps the root-finder from wandering out of the clipped region
    # instead of silently returning a constant there.
    penalty = 1e3 * np.array([logp[0] - lp0, logp[1] - lp1])
    r0sq, sig0sq = np.exp(lp0), np.exp(lp1)
    try:
        s_n, s_h = P.sources(sig0sq)
        phi1, eta1 = rk4_monopole(P.nodes, P.h, r0sq, s_n, s_h)
        log_rho0 = np.log(sig0sq / (4.0 * np.pi * GN * r0sq))
        out = np.array([
            log_rho0 - phi1 - s_n[-1] - np.log(P.rho1),
            np.log(4.0 * np.pi / 3.0) + log_rho0 + 3.0 * np.log(P.r1)
            - eta1 - np.log(P.M1),
        ]) + penalty
        if not np.all(np.isfinite(out)):
            return np.array([1e6, 1e6])
        return out
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.array([1e6, 1e6])


def _jacobian(logp, P, eps=1e-6):
    J = np.empty((2, 2))
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        J[:, k] = (_residual(logp + d, P) - _residual(logp - d, P)) / (2.0 * eps)
    return J


def _newton(logp, P, tol=1e-11, max_iter=60):
    """Damped Newton with a backtracking line search on |R|^2."""
    logp = np.array(logp, float)
    for _ in range(max_iter):
        R = _residual(logp, P)
        if np.max(np.abs(R)) < tol:
            return logp, True
        try:
            step = np.linalg.solve(_jacobian(logp, P), -R)
        except np.linalg.LinAlgError:
            return logp, False
        if not np.all(np.isfinite(step)):
            return logp, False
        # Trust region in log space: a full Newton step can be enormous near a
        # fold, where det(J) -> 0 by construction.
        scale = np.max(np.abs(step))
        if scale > 2.0:
            step *= 2.0 / scale
        f0 = float(R @ R)
        lam, moved = 1.0, False
        for _ in range(30):
            cand = logp + lam * step
            Rc = _residual(cand, P)
            if np.all(np.isfinite(Rc)) and float(Rc @ Rc) < f0 * (1.0 - 1e-4 * lam):
                logp, moved = cand, True
                break
            lam *= 0.5
        if not moved:
            return logp, float(np.max(np.abs(_residual(logp, P)))) < 1e-8
    return logp, float(np.max(np.abs(_residual(logp, P)))) < tol


def solve_spherical(r1, rho1, M1, Phi_b=None, n_steps=200, n_gl=N_GL_DEFAULT,
                    half_range=True, n_ramp=16, trajectory=False, tol=1e-11,
                    verify=True, verify_rtol=1e-6, method="bracket"):
    """Solve the spherical isothermal Jeans model matched at r1.

    Parameters
    ----------
    r1, rho1, M1 : float
        Matching radius, and the outer halo's density and enclosed mass there.
    Phi_b : callable or None
        Baryon potential, Phi_b(r) or Phi_b(r, theta). None runs the
        parameter-free path, which needs no iteration.
    n_steps : int
        RK4 steps. Fixed, never adapted. Self-convergence against a 9600-step
        reference is 4.1e-6 at 200 steps and 3.3e-9 at 1200.
    method : {"bracket", "ramp", "ramp+schedule"}
        How the branch is selected.

        "bracket" (default) inverts the problem by a bracketed monotone search
        and performs no continuation at all. On the density-matched curve the
        matching residual runs monotonically from -log(3R) at u1 -> 0 up to
        log(R_fold/R) at the first fold, so it changes sign at most once on the
        physical sheet and bisection cannot reach another branch. Checked
        against exhaustive, path-independent root enumeration on 300 targets:
        236 both found the same root to a worst relative difference of 3.9e-12,
        64 agreed there was none, 0 disagreements, and no target had more than
        one root on the physical sheet.

        It is also the cheapest option -- median 0.018 s against 0.044 s for the
        ramp and 0.128 s for the ramp with the schedule screen -- and the only
        one that covers R > R_MAX, where the ramp has no starting point at all.

        "ramp" runs the continuation and then the exact branch certificate.
        "ramp+schedule" is the earlier behaviour, kept for comparison; its
        screen is a filter rather than a proof and costs a second solve.
    n_ramp : int
        Continuation stages for the "ramp" methods. No number of stages fixes
        R > R_MAX: the first stage IS the baryon-free problem, which has no
        solution there, so those configurations are lost rather than wrong.
        Over 2889 fold-prior draws that do have a solution the ramp reached
        6.2% of the R > R_MAX cases, and n_ramp=128 did no better.
    verify : bool
        Screen the result of a "ramp" method -- the exact branch certificate
        for "ramp", schedule independence for "ramp+schedule". Ignored by
        "bracket", which is constructive and has nothing left to check.

    Returns
    -------
    SphericalResult
    """
    ratio = universal.matching_ratio_of(r1, rho1, M1)

    # ---- no baryons: interpolation, no root-find -------------------------
    if Phi_b is None:
        out = universal.solve(r1, rho1, M1, GN=GN)
        if out is None:
            return SphericalResult(np.nan, np.nan, False, "no_solution_R_above_Rmax",
                                   ratio=ratio)
        r0, sigma0 = out
        res = SphericalResult(r0, sigma0, True, "ok", 0.0, ratio, 0,
                              rho0=sigma0 ** 2 / (4.0 * np.pi * GN * r0 ** 2))
        if trajectory:
            _attach(res, r1, None, n_steps, n_gl, half_range)
        return res

    # ---- with baryons ----------------------------------------------------
    P = _Problem(r1, rho1, M1, n_steps, Phi_b, n_gl, half_range)

    if method == "bracket":
        br = branch.solve_bracketed(P)
        res = SphericalResult(
            br.r0, br.sigma0, br.success, br.reason, np.nan, ratio, br.n_eval,
            rho0=(br.sigma0 ** 2 / (4.0 * np.pi * GN * br.r0 ** 2)
                  if br.success else np.nan))
        if br.success:
            res.residual = float(np.max(np.abs(_residual(
                np.array([2.0 * np.log(br.r0), 2.0 * np.log(br.sigma0)]), P))))
            if trajectory:
                _attach(res, r1, Phi_b, n_steps, n_gl, half_range)
        return res
    if method not in ("ramp", "ramp+schedule"):
        raise ValueError("method must be 'bracket', 'ramp' or 'ramp+schedule';"
                         " got %r" % (method,))

    # seed on the universal curve, then ramp
    logp = universal.seed(r1, rho1, M1, GN=GN)
    if logp is None:
        return SphericalResult(np.nan, np.nan, False, "no_seed", ratio=ratio)

    ok = True
    prev = None
    for k in range(1, n_ramp + 1):
        P.set_upsilon(k / n_ramp)
        # Secant predictor: the solution path in Upsilon is smooth away from a
        # fold, so extrapolating the previous step roughly halves the Newton
        # iterations needed per stage.
        start = logp if prev is None else logp + (logp - prev)
        prev = logp
        logp, ok = _newton(start, P, tol=tol)
        if not ok:
            logp, ok = _newton(prev, P, tol=tol)
        if not ok:
            return SphericalResult(
                float(np.exp(0.5 * logp[0])), float(np.exp(0.5 * logp[1])),
                False, "fold_at_upsilon=%.4f" % P.upsilon,
                float(np.max(np.abs(_residual(logp, P)))), ratio, P.n_eval)

    residual = float(np.max(np.abs(_residual(logp, P))))
    r0 = float(np.exp(0.5 * logp[0]))
    sigma0 = float(np.exp(0.5 * logp[1]))
    reason = "ok" if ok else "not_converged"

    if ok and verify and method == "ramp+schedule":
        agree, other = _schedule_independent(
            r1, rho1, M1, Phi_b, n_steps, n_gl, half_range, n_ramp, tol,
            r0, verify_rtol)
        if not agree:
            ok = False
            reason = "schedule_dependent_root(r0=%.6g at n_ramp=%d)" % (
                other, 2 * n_ramp)
    elif ok and verify:
        # The exact certificate: free whenever the root is inside the
        # fast-path bound, otherwise it walks the matched curve down to the
        # regular core and requires the determinant sign to hold.
        Pc = _Problem(r1, rho1, M1, n_steps, Phi_b, n_gl, half_range)
        good, why = branch.certify(Pc, r0, sigma0)
        P.n_eval += Pc.n_eval
        if not good:
            ok = False
            reason = "off_physical_sheet(%s)" % why

    res = SphericalResult(r0, sigma0, ok, reason,
                          residual, ratio, P.n_eval,
                          rho0=sigma0 ** 2 / (4.0 * np.pi * GN * r0 ** 2))
    if trajectory:
        _attach(res, r1, Phi_b, n_steps, n_gl, half_range)
    return res


def _schedule_independent(r1, rho1, M1, Phi_b, n_steps, n_gl, half_range,
                          n_ramp, tol, r0, rtol):
    """Re-solve at twice the ramp density; return (agrees, the other r0).

    Past a saddle-node fold the residual stops being a correctness test: the
    spurious roots satisfy it to 1e-15, and in measured cases the answer was
    wrong by factors of 50 to 2000 while every other diagnostic looked clean.
    What distinguishes them is that they are artefacts of the path the
    continuation took, so they move when the schedule changes. A genuine root
    does not: over configurations with a real solution the two densities agree
    to far better than rtol.

    This is a necessary condition rather than a sufficient one -- two schedules
    can land on the same wrong branch -- so it is a filter, not a proof.
    """
    P = _Problem(r1, rho1, M1, n_steps, Phi_b, n_gl, half_range)
    logp = universal.seed(r1, rho1, M1, GN=GN)
    if logp is None:
        return True, np.nan          # nothing to compare against
    k_max = 2 * n_ramp
    prev = None
    for k in range(1, k_max + 1):
        P.set_upsilon(k / k_max)
        start = logp if prev is None else logp + (logp - prev)
        prev = logp
        logp, good = _newton(start, P, tol=tol)
        if not good:
            logp, good = _newton(prev, P, tol=tol)
        if not good:
            # The denser schedule failed where the sparser one succeeded. That
            # is itself evidence the root is not robust, so do not pass it.
            return False, np.nan
    other = float(np.exp(0.5 * logp[0]))
    return abs(other / r0 - 1.0) <= rtol, other


def _attach(res, r1, Phi_b, n_steps, n_gl, half_range):
    """Re-integrate at the solution to store the full (phi, eta) trajectory."""
    P = _Problem(r1, 1.0, 1.0, n_steps, Phi_b, n_gl, half_range)
    s_n, s_h = P.sources(res.sigma0 ** 2)
    phi, eta = rk4_monopole_full(P.nodes, P.h, res.r0 ** 2, s_n, s_h)
    res.r_nodes, res.phi, res.eta = P.nodes, phi, eta
