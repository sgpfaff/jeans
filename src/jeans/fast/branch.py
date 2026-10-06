"""Branch selection for the spherical solver, by bracketing rather than by path.

The matching problem is multi-valued. Writing u = r/r0, the no-baryon interior
curve has g = phi - eta turning over at u = 22.5442063, again at 244.85591 and
again at 2634.96690, with the matching ratio R = (1/3) exp(g) taking the values
1.26152659, 0.93199691 and 1.02169814 there and converging to 1. So a given R
in roughly (0.93, 1.26) has several roots, and they are all genuine: past a
fold the residual stops being a correctness test. A measured example, fixed in
tests/test_branch.py, has the 16-stage ramp returning a root 201 times too
small with a residual of 2.1e-13.

None of that is new. This is the classical isothermal spiral -- R is the
reciprocal of the Milne homology variable u_M, the folds are exactly the
classical locus u_M + v_M = 3, and the turning points space out as
exp(2 pi / sqrt 7) = 10.7483 (measured ratios 10.86, 10.76, 10.75). For the
dark-matter-only case the consequence is already in the literature: Robertson
et al. 2021 (arXiv:2009.07844, MNRAS 501, 4610), Appendix A, note that their
F(x) = rho/<rho> oscillates and that "this would lead to multiple possible r0",
arising once r1 > 1.78 r_s. What is added here is the criterion with baryons
and a solve that cannot land on the wrong sheet.

The relevant object is the map

    (u1, Lam) -> (R, mu),    u1 = r1/r0,  Lam = G Md / (sigma0^2 r0),
                             R  = M1/(4 pi r1^3 rho1),  mu = Md/(4 pi r1^3 rho1)

which is explicit: one integration, no root-find. Its folds are exactly where
det d(log R, log mu)/d(log u1, log Lam) vanishes, and those fold curves cut the
domain into sheets. The physical sheet is the one containing u1 -> 0, the
regular core. Sheets alternate in sign(det), so the sign at a root is necessary
but NOT sufficient -- sheets 1 and 3 share it, and the measured spurious roots
sit on sheet 3.

Three facts make this practical. The first two are theorems, the third is not.

  (1) At fixed u1 the density residual is monotone in sigma0^2, so the inner
      solve brackets.

  (2) Along the resulting matched curve the matching residual is monotone in
      u1 up to the fold, running from -log(3R) at u1 -> 0 up to log(R_fold/R).
      This follows from dlogR/dlog u1 = det / (dlog mu / dlog Lam), both
      factors being positive below the first fold. So the outer solve brackets
      too, and a sign change exists if and only if

          1/3 < R < R_fold(mu; shape),

      the exact existence criterion with baryons and the generalisation of
      R < R_MAX. Both facts are stated in the solver's own variables and so do
      not assume a Miyamoto-Nagai disc; tests/test_branch.py checks them for a
      Hernquist bulge as well.

  (3) The first fold is usually, but NOT always, outward of the no-baryon one.
      See U_SAFE.

A bracketed monotone search cannot land on the wrong sheet, so solve_bracketed
needs no continuation at all. Checked against exhaustive, path-independent root
enumeration on 300 targets: 236 found the same root to a worst relative
difference of 3.9e-12, 64 agreed there was none, 0 disagreements, and no target
had more than one root on the physical sheet. It is also the cheapest option
(median 0.018 s, against 0.044 s for the ramp and 0.128 s for the ramp with a
schedule-independence screen) and the only one that covers R > R_MAX.
"""
import numpy as np

from . import universal
from .kernels import rk4_monopole

__all__ = ["U_SAFE", "R_FLOOR", "BracketResult", "certify", "fold_u1",
           "solve_bracketed", "exists_with_baryons"]

GN = 4.302e-6

# The no-baryon fold. Quoted below universal.U_AT_R_MAX = 22.5442063 so that
# the fast path cannot claim a root sitting exactly on the fold, and below
# 22.544144, which is where a 200-step integration puts that fold -- coarse
# discretisation moves it inward by 6.2e-5.
#
# u1 < U_SAFE is NOT a theorem. The first fold can move inward of the no-baryon
# value: a Plummer sphere at b/r1 = 3 and Lam = 1000 puts it at 22.33819, and a
# thin shell at 0.6 r1 at 22.37768 (both converged against n_steps and verified
# in two independent integrators). To first order in baryon amplitude the fold
# shift is a linear functional of the baryon mass distribution, with a shell
# kernel that is POSITIVE inside 0.425 r1 and NEGATIVE between 0.425 r1 and r1,
# and zero beyond -- sign-indefinite, so no floor theorem exists. Mass spread
# across the thermalised region moves the fold inward; mass concentrated well
# inside it moves the fold outward.
#
# What makes the fast path sound in practice is that every counterexample found
# -- by a wide shape search, by an adversarial two-shell search, and by a
# first-order argument that the fold kernel and the R-shift kernel are never
# both negative -- requires R above R_MAX. So certify() applies the fast path
# only when R < R_MAX and falls through to the exact scan otherwise. That
# matters here and not upstream, because solve_bracketed does reach R > R_MAX,
# which no continuation can.
U_SAFE = 22.5441

# R -> 1/3 as u1 -> 0, for any Phi_b: phi, eta and s all vanish at the origin.
R_FLOOR = 1.0 / 3.0


class BracketResult:
    """Outcome of a bracketed solve, with the bracket that produced it."""

    __slots__ = ("r0", "sigma0", "success", "reason", "u1", "n_eval",
                 "u1_fold", "R", "R_fold")

    def __init__(self, r0=np.nan, sigma0=np.nan, success=False, reason="",
                 u1=np.nan, n_eval=0, u1_fold=np.nan, R=np.nan, R_fold=np.nan):
        self.r0, self.sigma0 = r0, sigma0
        self.success, self.reason = success, reason
        self.u1, self.n_eval = u1, n_eval
        self.u1_fold, self.R, self.R_fold = u1_fold, R, R_fold

    def __repr__(self):
        return ("BracketResult(r0=%.6g, sigma0=%.6g, success=%r, reason=%r, "
                "u1=%.4f, n_eval=%d)" % (self.r0, self.sigma0, self.success,
                                         self.reason, self.u1, self.n_eval))


# --------------------------------------------------------------------------
# the two residuals, in the solver's own variables
# --------------------------------------------------------------------------
def _integrate(P, r0, sig0sq):
    """(phi1, eta1, s1) at the matching radius."""
    P.n_eval += 1
    s_n, s_h = P.sources(sig0sq)
    phi1, eta1 = rk4_monopole(P.nodes, P.h, r0 * r0, s_n, s_h)
    return phi1, eta1, float(s_n[-1])


def _f_density(P, r0, sig0sq):
    """Density-match residual. Monotone increasing in sigma0^2 at fixed r0.

    As sigma0^2 -> inf the baryon source dies and log rho0 -> +inf; as
    sigma0^2 -> 0 both log rho0 and -s1 run to -inf. So the root is bracketed
    by expansion and is unique.
    """
    phi1, _, s1 = _integrate(P, r0, sig0sq)
    log_rho0 = np.log(sig0sq / (4.0 * np.pi * GN * r0 * r0))
    return log_rho0 - phi1 - s1 - np.log(P.rho1)


def _f_match(P, r0, sig0sq):
    """Matching residual, phi1 + s1 - eta1 - log(3R). Monotone in u1 up to the fold."""
    phi1, eta1, s1 = _integrate(P, r0, sig0sq)
    R = P.M1 / (4.0 * np.pi * P.r1 ** 3 * P.rho1)
    return phi1 + s1 - eta1 - np.log(3.0 * R)


def _sigma_for(P, r0, guess=None, tol=1e-12, max_expand=200, max_iter=200):
    """Solve the density match for sigma0^2 at fixed r0, by bracketing.

    Returns log(sigma0^2), or None if no bracket was found or the iteration did
    not converge. Never returns an unconverged iterate: an earlier version
    ended with ``return 0.5 * (a + b)`` after the loop, which handed back a
    non-root whenever false position stagnated, and the caller then read a
    matching residual off the wrong point on the curve.

    The residual is monotone increasing in sigma0^2 at fixed r0 -- as
    sigma0^2 -> inf the baryon source dies and log rho0 -> +inf, as
    sigma0^2 -> 0 both log rho0 and -s1 run to -inf -- so the root is bracketed
    by expansion and is unique. Verified by direct scan: one sign change over
    log(sigma0^2) in [2, 30] at u1 = 0.14, 1.0 and 4.56 for a disc at mu = 8.

    Iteration is Illinois rather than plain false position. The residual is very
    flat on one side for a deep disc, which pins an endpoint and makes the
    unmodified secant creep; halving the retained endpoint's value restores
    superlinear convergence.

    `guess` warm-starts the expansion from the previous point on the matched
    curve, which is worth about a factor of five.
    """
    l = np.log(GN * P.M1 / P.r1) if guess is None else float(guess)
    a = b = l
    fa = fb = _f_density(P, r0, np.exp(l))
    if not np.isfinite(fa):
        return None
    step = 1.0
    for _ in range(max_expand):
        if (fa < 0.0) != (fb < 0.0):
            break
        if fa > 0.0:
            b, fb = a, fa
            a -= step
            fa = _f_density(P, r0, np.exp(a))
        else:
            a, fa = b, fb
            b += step
            fb = _f_density(P, r0, np.exp(b))
        step *= 1.6
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None
    else:
        return None
    if fa > 0.0:                       # orient so that fa < 0 < fb
        a, b, fa, fb = b, a, fb, fa

    side = 0
    for _ in range(max_iter):
        if abs(b - a) < tol:
            return 0.5 * (a + b)
        m = 0.5 * (a + b)
        if fb != fa:
            sc = a - fa * (b - a) / (fb - fa)
            if min(a, b) < sc < max(a, b):
                m = sc
        fm = _f_density(P, r0, np.exp(m))
        if not np.isfinite(fm):
            return None
        if abs(fm) < 1e-12:
            return m
        if fm < 0.0:
            a, fa = m, fm
            if side == -1:
                fb *= 0.5
            side = -1
        else:
            b, fb = m, fm
            if side == +1:
                fa *= 0.5
            side = +1
    return None                        # did not converge: say so, do not guess


def _g(P, r0, guess=None):
    """The matching residual on the density-matched curve. None if unbracketable.

    A warm start that is far from the answer can fail to bracket, so a failed
    warm solve is retried cold rather than reported as a failure: along the
    matched curve log(sigma0^2) spans more than twelve e-folds between u1 = 1e-3
    and the root, and the secant can jump most of that in one step.
    """
    ls = _sigma_for(P, r0, guess=guess)
    if ls is None and guess is not None:
        ls = _sigma_for(P, r0, guess=None)
    if ls is None:
        return None, None
    return _f_match(P, r0, np.exp(ls)), ls


# --------------------------------------------------------------------------
# the fold, and the certificate
# --------------------------------------------------------------------------
def _det(P, r0, sig0sq, eps=1e-5):
    """det d(log R, log mu)/d(log u1, log Lam), in the solver's variables.

    The change of variables from (log r0^2, log sigma0^2) to
    (log u1, log Lam) is linear with a constant determinant, so the SIGN
    computed here is the sheet label regardless of which pair is used.
    """
    out = np.empty((2, 2))
    for k, (dr, ds) in enumerate(((eps, 0.0), (0.0, eps))):
        pp, ep, sp = _integrate(P, r0 * np.exp(dr), sig0sq * np.exp(ds))
        pm, em, sm = _integrate(P, r0 * np.exp(-dr), sig0sq * np.exp(-ds))
        out[0, k] = ((pp - ep + sp) - (pm - em + sm)) / (2.0 * eps)
        # log mu = log Lam - 3 log u1 + phi + s. Rewritten in the variables
        # actually perturbed here, (log r0, log sigma0^2), that explicit part
        # is 2 log r0 - log sigma0^2 plus a constant, hence +2 and -1. The
        # change of variables has determinant 1, so the sign is unchanged.
        out[1, k] = ((pp + sp) - (pm + sm)) / (2.0 * eps) + (2.0 if k == 0 else -1.0)
    return float(out[0, 0] * out[1, 1] - out[0, 1] * out[1, 0])


def fold_u1(P, lo=None, hi=400.0, n_scan=48, n_bisect=40):
    """Locate the first fold in u1 along the density-matched curve.

    Returns (u1_fold, R_fold) or (nan, nan) if no fold was found below `hi`.
    Used for the exact existence criterion and as the outer bracket.
    """
    lo = U_SAFE if lo is None else lo
    us = np.geomspace(lo, hi, n_scan)
    prev_u, prev_d, warm = None, None, None
    for u in us:
        ls = _sigma_for(P, P.r1 / u, guess=warm)
        if ls is None:
            break
        warm = ls
        d = _det(P, P.r1 / u, np.exp(ls))
        if not np.isfinite(d):
            break
        if prev_d is not None and np.sign(d) != np.sign(prev_d):
            a, b = prev_u, u
            for _ in range(n_bisect):
                m = np.sqrt(a * b)
                ls = _sigma_for(P, P.r1 / m, guess=warm)
                if ls is None:
                    break
                warm = ls
                dm = _det(P, P.r1 / m, np.exp(ls))
                if np.sign(dm) == np.sign(prev_d):
                    a = m
                else:
                    b = m
            uf = np.sqrt(a * b)
            gm, _ = _g(P, P.r1 / uf, guess=warm)
            R = P.M1 / (4.0 * np.pi * P.r1 ** 3 * P.rho1)
            return uf, (np.nan if gm is None else float(np.exp(gm) * R))
        prev_u, prev_d = u, d
    return np.nan, np.nan


def certify(P, r0, sigma0, u_safe=U_SAFE):
    """Is this converged root on the physical sheet? Returns (ok, reason).

    Fast path: u1 < u_safe together with R < R_MAX. The bound on u1 alone is
    not a proof -- see U_SAFE -- but every configuration found that breaks it
    needs R above R_MAX, so the pair is sound wherever it fires and costs two
    divisions. Otherwise the matched curve is walked from the root down to
    u1 -> 0 and the determinant sign is required to hold throughout, which
    places the root in the same connected component as the regular core. That
    walk is exact up to its own resolution; it cannot see a fold pair narrower
    than its step.
    """
    u1 = P.r1 / r0
    if not np.isfinite(u1) or u1 <= 0.0:
        return False, "non_finite_u1"
    R = P.M1 / (4.0 * np.pi * P.r1 ** 3 * P.rho1)
    if u1 < u_safe and R < universal.R_MAX:
        return True, "u1<%.4f,R<R_MAX" % u_safe
    d_root = _det(P, r0, sigma0 ** 2)
    if not np.isfinite(d_root) or d_root <= 0.0:
        return False, "det<=0_at_root(u1=%.3f)" % u1
    warm = 2.0 * np.log(sigma0)
    for u in np.geomspace(u1 * 0.999, 0.05, 40):
        ls = _sigma_for(P, P.r1 / u, guess=warm)
        if ls is None:
            return False, "unbracketable_at_u1=%.3f" % u
        warm = ls
        if _det(P, P.r1 / u, np.exp(ls)) <= 0.0:
            return False, "fold_between_core_and_root(u1=%.3f)" % u
    return True, "det>0_down_to_core(u1=%.3f)" % u1


# --------------------------------------------------------------------------
# the bracketed solve
# --------------------------------------------------------------------------
def exists_with_baryons(P, hi=400.0):
    """Exact existence test: 1/3 < R < R_fold. Returns (exists, R, R_fold)."""
    R = P.M1 / (4.0 * np.pi * P.r1 ** 3 * P.rho1)
    if R <= R_FLOOR:
        return False, R, np.nan
    _, R_fold = fold_u1(P, hi=hi)
    if not np.isfinite(R_fold):
        return False, R, R_fold
    return bool(R < R_fold), R, R_fold


def solve_bracketed(P, u_lo=1e-3, u_hi=None, tol=1e-12, max_bisect=200):
    """Solve by bracketing on the physical sheet. Cannot select another branch.

    The matching residual runs monotonically from -log(3R) at u1 -> 0 to
    log(R_fold/R) at the fold, so it changes sign at most once on the sheet and
    bisection converges to the physical root or proves there is none.
    """
    P.n_eval = 0
    R = P.M1 / (4.0 * np.pi * P.r1 ** 3 * P.rho1)
    if not np.isfinite(R) or R <= R_FLOOR:
        return BracketResult(reason="R<=1/3_no_solution", R=R, n_eval=P.n_eval)

    g_lo, warm = _g(P, P.r1 / u_lo)
    R_fold = np.nan
    if u_hi is None:
        # The fold search is the expensive part, and it is almost never needed:
        # the residual is already positive at U_SAFE whenever the root lies
        # inward of it, which is the overwhelming majority of cases. Only when
        # it is still negative there does the bracket have to be extended past
        # the fast-path bound, and only then is the fold located.
        g_safe, w2 = _g(P, P.r1 / U_SAFE, guess=warm)
        if g_safe is not None and g_safe > 0.0:
            u_hi, g_hi = U_SAFE, g_safe
        else:
            u_hi, R_fold = fold_u1(P)
            if not np.isfinite(u_hi):
                u_hi, R_fold = 400.0, np.nan
            g_hi, w2 = _g(P, P.r1 / (u_hi * (1.0 - 1e-9)), guess=w2)
    else:
        g_hi, w2 = _g(P, P.r1 / (u_hi * (1.0 - 1e-9)), guess=warm)
    if g_lo is None or g_hi is None:
        return BracketResult(reason="unbracketable", R=R, R_fold=R_fold,
                             u1_fold=u_hi, n_eval=P.n_eval)
    if g_lo > 0.0:
        return BracketResult(reason="R_below_core_limit", R=R, R_fold=R_fold,
                             u1_fold=u_hi, n_eval=P.n_eval)
    if g_hi < 0.0:
        return BracketResult(reason="R_above_fold_no_solution", R=R,
                             R_fold=R_fold, u1_fold=u_hi, n_eval=P.n_eval)

    # Illinois, not plain false position. The matching residual is flat on the
    # left -- it sits within 1e-3 of -log(3R) for every u1 below about 0.1 --
    # so an unmodified secant retains the left endpoint forever and the bracket
    # creeps. Halving the retained endpoint's value on a repeated side restores
    # superlinear convergence. Before this, deep-disc cases exited on the
    # iteration cap at u1 = 0.14 while the true root was at 4.56, and reported
    # success.
    a, b, fa, fb = u_lo, u_hi, g_lo, g_hi
    side = 0
    converged = False
    for _ in range(max_bisect):
        if b - a < tol * a:
            converged = True
            break
        m = np.sqrt(a * b)
        if fb != fa:
            sc = np.exp(np.log(a) - fa * (np.log(b) - np.log(a)) / (fb - fa))
            if a * (1 + 1e-12) < sc < b * (1 - 1e-12):
                m = sc
        fm, w = _g(P, P.r1 / m, guess=warm)
        if fm is None:
            return BracketResult(reason="inner_solve_failed_at_u1=%.4g" % m, R=R,
                                 R_fold=R_fold, u1_fold=u_hi, n_eval=P.n_eval)
        warm = w
        if abs(fm) < 1e-13:
            a = b = m
            converged = True
            break
        if fm < 0.0:
            a, fa = m, fm
            if side == -1:
                fb *= 0.5
            side = -1
        else:
            b, fb = m, fm
            if side == +1:
                fa *= 0.5
            side = +1
    if not converged:
        return BracketResult(reason="bisection_did_not_converge", R=R,
                             R_fold=R_fold, u1_fold=u_hi, n_eval=P.n_eval)
    u1 = np.sqrt(a * b)
    r0 = P.r1 / u1
    ls = _sigma_for(P, r0, guess=warm)
    if ls is None:
        return BracketResult(reason="final_inner_solve_failed", R=R,
                             R_fold=R_fold, u1_fold=u_hi, n_eval=P.n_eval)
    return BracketResult(r0=float(r0), sigma0=float(np.exp(0.5 * ls)),
                         success=True, reason="bracketed", u1=float(u1),
                         n_eval=P.n_eval, u1_fold=float(u_hi), R=float(R),
                         R_fold=float(R_fold))
