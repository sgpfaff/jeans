"""Axisymmetric solver: nonlinear monopole, linear shape.

The package relaxes all 2(L_max/2 + 1) fields simultaneously on a staggered
grid. The structural fact that makes a reduction possible is that the baryon
sector and the shape sector need different treatments:

  * the baryon sector is not perturbative -- a Milky-Way-like disc sits at a
    depth where a second-order expansion is still 17% wrong -- so the monopole
    is solved nonlinearly, with continuation in the baryon amplitude;

  * the halo's own multipoles stay small, so linearising the L>0 sector about
    the spherical background is accurate to the square of that, and the shape
    follows from one particular plus one homogeneous march per mode.

How accurate, measured rather than asserted. The reference is the package's
own relaxation, which builds its source as the exact angular integral of
exp(-phi_b - sum_L phi_L Z_L) and Newton-iterates on it, so it assumes no
linearity. Over 150 configurations with q0 swept from 0.95 down to 0.25,
spanning max|phi_2| from 0.02 to 0.94:

    max|phi_2|      n    median err r0    median err phi_2
    [0.00, 0.05)    6        1.6e-4            1.1e-3
    [0.15, 0.20)   13        1.8e-4            1.7e-3
    [0.30, 0.40)   17        2.1e-4            3.0e-3
    [0.50, 0.70)   20        2.0e-4            5.0e-3
    [0.70, 0.94]    7        2.0e-4            7.3e-3

The monopole column is flat, and a resolution ladder shows why: refining both
sides together (r_grid = n_steps = 200, 400, 800, 1600) drives the r0
difference 1.61e-4 -> 3.82e-5 -> 7.68e-6 -> 6.52e-8 while the phi_2 difference
sits at 4.95e-4, 5.05e-4, 5.06e-4, 5.07e-4. So the flat 2e-4 in r0 was the
comparison's own discretisation, and the linearisation error in the solved r0
and sigma0 is below 1e-7 -- the monopole is indifferent to the shape
amplitude (measured correlation of the r0 error with |phi_2|: -0.03).

The shape carries the real error, and it is predictable: err/|phi_2| is
constant at 9.3e-3 (p10 4.4e-3, p90 1.4e-2), which is the signature of the
dropped quadratic term. At |phi_2| = 0.94 -- q0 = 0.25, flatter than any real
halo -- the shape is still right to 0.96%.

An earlier version of this docstring quoted "below 0.272 over 1080
configurations" as a validated range. That was the largest value in a sample,
not a boundary: 35% of a realistic prior exceeds it, and exceeding it costs
nothing measurable in r0 and under 1% in the shape. See bench/shape/.

L=4 is different in kind, and the difference is structural rather than a
matter of resolution. Over 42 configurations with L = (0, 2, 4) the relative
error in phi_4 is 1.7% to 6.9% (median 3.3%), it does NOT shrink as the halo
rounds, and it does not move at all under refinement: at q0 = 0.8 it reads
3.20e-2, 3.22e-2, 3.23e-2 at n_steps = 200, 400, 800, and is identical at
n_outer = 4 and 12 to every printed digit, while the r0 error over the same
ladder falls 2.36e-4 -> 6.68e-5 -> 2.48e-5. So it is neither discretisation
nor an unconverged alternation. It is this solver's diagonal approximation:
each L is marched independently, which drops the 2<->4 coupling, and that
coupling supplies a larger share of phi_4 for rounder halos (the error is
3.2% at q0 = 0.8 against 1.7% at q0 = 0.5, where the direct J_4 drive is
stronger). Including the off-diagonal block would fix it, which is the same
block-tridiagonal solve that L=6 needs.

Read it as a relative error on a small quantity, not as a reason to drop L=4.
phi_4 runs 10 to 20 times smaller than phi_2, so the ABSOLUTE contribution is
about as accurate: at q0 = 0.5 with |phi_2| = 0.456 and |phi_4| = 0.0347 the
two absolute errors are 1.7e-3 and 1.1e-3. Including L=4 still improves the
density; what should not be done is quoting phi_4 itself as a precise
amplitude.

Linearising is required rather than merely convenient. Nonlinear shooting in
the multipole amplitudes needs a ~5e4 cancellation against the homogeneous r^L
mode at L=4, loses about four digits, and fails to converge at all at L=6.

Variables follow the package exactly, so results are directly comparable:

    dphi_L/dr = mu_L / r^2
    dmu_L/dr  = L(L+1) phi_L + 4 pi (r^2 / r0^2) A_L(r)
    A_L(r)    = <Z_L exp(-phi_b - phi_dm)>,  phi_dm = sum_L phi_L Z_L

with boundary conditions phi_L(0) = 0, mu_00(0) = 0,
rho0 A_0(r1)/Z_0 = rho1, mu_00(r1) = sqrt(4 pi) G M1 / sigma0^2, and for L>0
mu_L(r1) + r1 (L+1) phi_L(r1) + 4 pi G r1^(L+1) J_L / sigma0^2 = 0.
"""
import numpy as np

from . import universal
from .kernels import (rk4_monopole_full, rk4_multipole_linear,
                      rk4_multipole_trajectory, source_from_grid)
from .quadrature import N_GL_DEFAULT, gauss_legendre, harmonics, tabulate_baryons
from .solver import GN, _Problem, _newton, _residual

__all__ = ["Result2D", "solve_axisymmetric", "PSI_VALID", "PSI_HARD"]

# The quantity linearised is psi = sum_{L>0} phi_L Z_L, the L>0 part of the
# exponent in A_L = <Z_L exp(-phi_b - phi_dm)>, so a bound belongs on psi and
# not on the coefficient phi_L. For L=2 alone max|psi| = 0.6208 max|phi_2| on
# the solver's own grid (sqrt(5/4pi) = 0.6308 in the continuum; the nodes do
# not include theta = 0).
#
# PSI_VALID is where the measured scatter puts the shape error at 1%, using
# the p90 of err/|phi_2| = 1.44e-2 rather than its median. Nothing in a
# realistic prior comes close: the largest max|psi| over 230 converged
# configurations with q0 down to 0.4 was 0.37.
PSI_VALID = 0.43

# Beyond this the error curve has never been measured, so the solver refuses
# rather than extrapolating. It is roughly twice the most extreme
# configuration reached (max|phi_2| = 0.94 at q0 = 0.25, i.e. psi = 0.58).
PSI_HARD = 1.2

# err/|phi_2| from the same measurement, used for the reported estimate.
SHAPE_ERR_PER_PHI = 9.3e-3

_Z00 = 1.0 / np.sqrt(4.0 * np.pi)


class Result2D:
    """Solution of the axisymmetric model. phi_L are the package's multipoles."""

    def __init__(self, r0, sigma0, L_list, phi_L, r_nodes, success, reason,
                 residual, ratio, n_eval, lag=np.nan, max_psi=np.nan):
        self.r0 = r0
        self.sigma0 = sigma0
        self.L_list = list(L_list)
        self.phi_L = phi_L          # dict: L -> array over r_nodes
        self.r_nodes = r_nodes
        self.success = success
        self.reason = reason
        self.residual = residual
        # Newton residual and alternation lag are different numbers and were
        # previously conflated. The outer loop installs pass-k shape feedback
        # as its last statement, so a residual evaluated after the loop scores
        # logp -- which solves the pass-(k-1) equations -- against the pass-k
        # ones. That measures how much the alternation has left to go, not
        # whether Newton converged. Reported separately: `residual` is the
        # Newton residual at the solve that produced r0, `lag` is the
        # alternation lag. Measured for one case at n_outer = 1/2/3/4, the lag
        # runs 1.68e-4, 6.79e-8, 2.98e-11, 1.24e-14.
        self.lag = lag
        # max |psi| over the solver's own (r, theta) grid, where
        # psi = sum_{L>0} phi_L Z_L is the quantity the shape sector
        # linearises. Always reported, so a caller can see where on the
        # measured error curve its answer sits instead of having to guess.
        self.max_psi = max_psi
        # An estimate, not a bound: the measured err/|phi_2| has median
        # 9.3e-3 with p10 4.4e-3 and p90 1.4e-2, so treat this as right to a
        # factor of about 1.5 either way. It applies to the SHAPE; the error
        # the linearisation puts into r0 and sigma0 is below 1e-7 and is not
        # worth reporting.
        self.shape_rel_error = (SHAPE_ERR_PER_PHI * max_psi / 0.6208
                                if np.isfinite(max_psi) else np.nan)
        self.linear_response_ok = (bool(max_psi <= PSI_VALID)
                                   if np.isfinite(max_psi) else False)
        self.ratio = ratio
        self.n_residual_evals = n_eval
        self.rho0 = sigma0 ** 2 / (4.0 * np.pi * GN * r0 ** 2)

    def phi_at(self, L, r):
        return np.interp(r, self.r_nodes, self.phi_L[L])

    def rho(self, r, theta):
        """Dark-matter density at (r, theta), for r <= r1."""
        Z = harmonics(self.L_list, n=1, half_range=False)  # shape only; recompute below
        x = np.cos(theta)
        tot = np.zeros(np.shape(r * np.ones_like(x, dtype=float)), dtype=float)
        from scipy.special import eval_legendre
        for L in self.L_list:
            ZL = np.sqrt((2 * L + 1) / (4.0 * np.pi)) * eval_legendre(L, x)
            tot = tot + self.phi_at(L, r) * ZL
        return self.rho0 * np.exp(-tot)

    def __repr__(self):
        return (f"Result2D(r0={self.r0:.6g}, sigma0={self.sigma0:.6g}, "
                f"L_list={self.L_list}, success={self.success}, "
                f"residual={self.residual:.2e}, lag={self.lag:.2e}, "
                f"max_psi={self.max_psi:.3f}, "
                f"shape_rel_error~{self.shape_rel_error:.1e})")


def _angular_terms(grid, w, Z, phi00, sig0sq, r0sq):
    """Source and diagonal-coupling coefficients for every mode, at every node.

    Returns (S, W) each of shape (n_L, n_r):

        S_L = (4 pi / r0^2) <Z_L exp(-phi_b - phi_00 Z_0)>
        W_L = (4 pi / r0^2) <Z_L^2 exp(-phi_b - phi_00 Z_0)>

    S_L for L>0 is evaluated as a difference against the angular mean. The
    quadrature satisfies sum_j w_j Z_L(x_j) = 0 identically for L>0, so forming
    <Z_L exp(-v)> directly is pure round-off whenever the anisotropy |v| is
    small -- which is exactly the inner-boundary regime, where phi_{L>0} is of
    order 1e-20. Computed naively the Jacobian columns for the shape amplitudes
    come out numerically zero and the solve does not move at all. Factoring the
    monopole out and using expm1 on the anisotropic part only is what makes the
    L>0 sector solvable.
    """
    pref = 4.0 * np.pi / r0sq
    v = grid / sig0sq                                   # (n_r, n_th)
    vref = v.min(axis=1, keepdims=True)
    e = np.exp(-(v - vref))                             # O(1), no overflow
    mono = np.exp(-vref[:, 0] - phi00 * _Z00)           # (n_r,)

    n_L = Z.shape[0]
    S = np.empty((n_L, grid.shape[0]))
    W = np.empty((n_L, grid.shape[0]))
    ebar = (w * e).sum(axis=1)                          # (n_r,)
    for k in range(n_L):
        ZL = Z[k]
        if k == 0:
            S[k] = pref * mono * (w * ZL * e).sum(axis=1)
        else:
            # sum_j w_j Z_L = 0, so subtract the angular mean before summing
            S[k] = pref * mono * (w * ZL * (e - ebar[:, None])).sum(axis=1)
        W[k] = pref * mono * (w * ZL * ZL * e).sum(axis=1)
    return S, W


def solve_axisymmetric(r1, rho1, M1, J_L=None, L_list=(0, 2), Phi_b=None,
                       n_steps=200, n_gl=N_GL_DEFAULT, half_range=True,
                       n_ramp=16, n_outer=4, tol=1e-11):
    """Solve the axisymmetric isothermal Jeans model matched at r1.

    Parameters
    ----------
    r1, rho1, M1 : float
        Matching radius and the outer halo's spherically averaged density and
        enclosed mass there.
    J_L : sequence or None
        Outer-halo potential moments, one per entry of L_list, as
        outer_halo.potential_moments returns them. None means a spherical outer
        halo, which drives no L>0 structure by itself.
    L_list : sequence of int
        Even multipoles, starting with 0.
    n_outer : int
        Passes over (monopole, shape). The sectors couple only weakly -- the
        monopole feels the shape at second order -- so this converges fast:
        the measured per-pass contraction is 3e-3 to 4e-4 away from a fold.
        The default is 4 rather than 3 because the third pass still leaves up
        to 4.0e-7 relative error in r0 (median 2.6e-9 over 150 realistic
        cases) and up to 2.6e-3 near a fold, where the contraction degrades
        toward 0.5; the fourth costs one extra Newton and buys three to four
        orders of magnitude.

    Returns
    -------
    Result2D
    """
    L_list = list(L_list)
    if L_list[0] != 0:
        raise ValueError("L_list must start with 0")
    if any(L % 2 for L in L_list):
        raise ValueError("odd L are not supported (z-symmetry assumed)")
    higher = L_list[1:]
    if J_L is None:
        J_L = np.zeros(len(L_list))
    J_L = np.asarray(J_L, float)
    if J_L.shape[0] != len(L_list):
        raise ValueError("J_L must have one entry per L in L_list")

    # ---- set up ---------------------------------------------------------
    P = _Problem(r1, rho1, M1, n_steps, Phi_b, n_gl, half_range)
    x, w = gauss_legendre(n_gl, half_range)
    Z = harmonics(L_list, n=n_gl, half_range=half_range)
    nodes, half, h = P.nodes, P.half, P.h

    # The monopole path collapses a theta-independent Phi_b to one column; the
    # shape sector needs the full angular grid, and so does the feedback term.
    if P.grid_n.shape[1] != n_gl:
        P.grid_n = np.repeat(P.grid_n[:, :1], n_gl, axis=1)
        P.grid_h = np.repeat(P.grid_h[:, :1], n_gl, axis=1)
        P.w = w
        P._isotropic = False
    P._cache.clear()

    logp = universal.seed(r1, rho1, M1, GN=GN)
    ratio = universal.matching_ratio_of(r1, rho1, M1)
    if logp is None:
        return Result2D(np.nan, np.nan, L_list, {}, nodes, False, "no_seed",
                        np.nan, ratio, 0)

    phi_L = {L: np.zeros(len(nodes)) for L in L_list}
    ok = True

    # ---- alternate: nonlinear monopole, then linear shape ---------------
    # The two sectors couple only weakly. The monopole feels the shape through
    # the effective angular average, which is second order in phi_{L>0}; the
    # shape feels the monopole through the background density. Three passes is
    # comfortably enough, and the Newton warm-starts each time.
    newton_res = np.nan
    max_psi = 0.0        # stays 0 when L_list == [0]
    for outer in range(max(1, n_outer)):

        # -- monopole --
        if Phi_b is None and outer == 0:
            P.set_upsilon(1.0)
            logp, ok = _newton(logp, P, tol=tol)
        elif outer == 0:
            prev = None
            for k in range(1, n_ramp + 1):
                P.set_upsilon(k / n_ramp)
                start_pt = logp if prev is None else logp + (logp - prev)
                prev = logp
                logp, ok = _newton(start_pt, P, tol=tol)
                if not ok:
                    logp, ok = _newton(prev, P, tol=tol)
                if not ok:
                    return Result2D(float(np.exp(0.5 * logp[0])),
                                    float(np.exp(0.5 * logp[1])), L_list, {},
                                    nodes, False,
                                    "fold_at_upsilon=%.4f" % P.upsilon,
                                    np.nan, ratio, P.n_eval)
        else:
            logp, ok = _newton(logp, P, tol=tol)
        if not ok:
            break

        # Scored against the feedback logp actually solved against, i.e.
        # before this pass installs its own. This is the Newton residual.
        newton_res = float(np.max(np.abs(_residual(logp, P))))

        r0sq, sig0sq = np.exp(logp[0]), np.exp(logp[1])
        s_n, s_h = P.sources(sig0sq)
        phi00_lin, _ = rk4_monopole_full(nodes, h, r0sq, s_n, s_h)
        phi00 = phi00_lin / _Z00          # package multipole normalisation
        phi_L[0] = phi00
        phi00_half = np.interp(half, nodes, phi00)

        if not higher:
            break

        # -- shape --
        S_n, W_n = _angular_terms(P.grid_n, w, Z, phi00, sig0sq, r0sq)
        S_h, W_h = _angular_terms(P.grid_h, w, Z, phi00_half, sig0sq, r0sq)
        for k, L in enumerate(L_list):
            if L == 0:
                continue
            src_n = nodes ** 2 * S_n[k]
            src_h = half ** 2 * S_h[k]
            pp, mp, ph, mh = rk4_multipole_linear(
                nodes, h, L, W_n[k], W_h[k], src_n, src_h)
            K = 4.0 * np.pi * GN * r1 ** (L + 1) * J_L[k] / sig0sq
            denom = mh + r1 * (L + 1) * ph
            c = -(mp + r1 * (L + 1) * pp + K) / denom if denom != 0.0 else 0.0
            phi_L[L] = rk4_multipole_trajectory(
                nodes, h, L, W_n[k], W_h[k], src_n, src_h, c)

        # -- feed the shape back into the monopole's angular average --
        extra_n = np.zeros_like(P.grid_n)
        extra_h = np.zeros_like(P.grid_h)
        for k, L in enumerate(L_list):
            if L == 0:
                continue
            extra_n += sig0sq * np.outer(phi_L[L], Z[k])
            extra_h += sig0sq * np.outer(np.interp(half, nodes, phi_L[L]), Z[k])
        # extra_n holds sigma0^2 * psi on the (r, theta) grid, so this is
        # max|psi| without recomputing the harmonics.
        max_psi = float(np.max(np.abs(extra_n))) / sig0sq
        P.set_shape_feedback(extra_n, extra_h)

    lag = float(np.max(np.abs(_residual(logp, P))))
    reason = "ok" if ok else "not_converged"
    if max_psi > PSI_HARD:
        # Takes precedence over a convergence failure: past this the shape
        # sector is far outside anything measured, so the alternation
        # struggling is a symptom rather than the cause, and reporting the
        # linearisation is the more useful answer.
        ok = False
        reason = ("outside_linear_response(max_psi=%.3f > %.3f)"
                  % (max_psi, PSI_HARD))
    return Result2D(float(np.exp(0.5 * logp[0])), float(np.exp(0.5 * logp[1])),
                    L_list, phi_L, nodes, bool(ok), reason, newton_res, ratio,
                    P.n_eval, lag=lag, max_psi=max_psi)
