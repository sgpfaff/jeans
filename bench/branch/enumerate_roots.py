"""Exhaustive, path-independent root enumeration for the reduced problem.

Ground truth must not share the failure mode it is testing.  A 120-stage ramp
is still a ramp: if it crosses a fold it is wrong in exactly the way the
16-stage ramp is wrong.  So the reference here never continues anything.  It
evaluates the explicit forward map on a dense (log u1, log Lam) grid, takes
every cell in which both residuals change sign, polishes each with Newton, and
deduplicates.  That finds every root in the box, including the ones no
continuation path would ever reach.
"""
import numpy as np
from numba import njit, prange

import fwd


@njit(cache=True, parallel=True)
def _grid(lu, lL, alpha1, beta1, n_steps, xs, ws):
    nu, nL = lu.shape[0], lL.shape[0]
    R = np.empty((nu, nL))
    M = np.empty((nu, nL))
    for i in prange(nu):
        u1 = np.exp(lu[i])
        for j in range(nL):
            phi1, eta1, s1 = fwd.integrate(u1, np.exp(lL[j]), alpha1, beta1,
                                           n_steps, xs, ws)
            R[i, j] = phi1 - eta1 + s1 - np.log(3.0)
            M[i, j] = lL[j] - 3.0 * lu[i] + phi1 + s1
    return R, M


def _F(x, logR_t, logmu_t, alpha1, beta1, n_steps, xs, ws):
    phi1, eta1, s1 = fwd.integrate(np.exp(x[0]), np.exp(x[1]), alpha1, beta1,
                                   n_steps, xs, ws)
    return np.array([phi1 - eta1 + s1 - np.log(3.0) - logR_t,
                     x[1] - 3.0 * x[0] + phi1 + s1 - logmu_t])


def _J(x, *a, eps=1e-6):
    J = np.empty((2, 2))
    for k in range(2):
        d = np.zeros(2); d[k] = eps
        J[:, k] = (_F(x + d, *a) - _F(x - d, *a)) / (2 * eps)
    return J


def polish(x0, args, tol=1e-12, it=80):
    x = np.array(x0, float)
    for _ in range(it):
        F = _F(x, *args)
        if not np.all(np.isfinite(F)):
            return None
        if np.max(np.abs(F)) < tol:
            return x
        try:
            step = np.linalg.solve(_J(x, *args), -F)
        except np.linalg.LinAlgError:
            return None
        sc = np.max(np.abs(step))
        if sc > 1.0:
            step *= 1.0 / sc
        f0 = F @ F
        lam = 1.0
        for _ in range(40):
            c = x + lam * step
            Fc = _F(c, *args)
            if np.all(np.isfinite(Fc)) and Fc @ Fc < f0 * (1 - 1e-4 * lam):
                x = c
                break
            lam *= 0.5
        else:
            return x if np.max(np.abs(_F(x, *args))) < 1e-9 else None
    return x if np.max(np.abs(_F(x, *args))) < 1e-9 else None


def all_roots(logR_t, logmu_t, alpha1, beta1, u_range=(0.2, 400.0),
              L_range=(1e-6, 200.0), nu=400, nL=320, n_steps=200, n_gl=16):
    """Every root of the reduced system in the box, with its det sign."""
    xs, ws = fwd.nodes(n_gl)
    lu = np.log(np.geomspace(*u_range, nu))
    lL = np.log(np.geomspace(*L_range, nL))
    R, M = _grid(lu, lL, alpha1, beta1, n_steps, xs, ws)
    A = R - logR_t
    B = M - logmu_t
    args = (logR_t, logmu_t, alpha1, beta1, n_steps, xs, ws)

    def corners(Z):
        return np.stack([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])
    ca, cb = corners(A), corners(B)
    ok = np.isfinite(ca).all(0) & np.isfinite(cb).all(0)
    cand = ok & (ca.min(0) <= 0) & (ca.max(0) >= 0) & (cb.min(0) <= 0) & (cb.max(0) >= 0)
    out = []
    for i, j in zip(*np.where(cand)):
        x0 = np.array([0.5 * (lu[i] + lu[i + 1]), 0.5 * (lL[j] + lL[j + 1])])
        x = polish(x0, args)
        if x is None:
            continue
        if not (lu[0] - 0.5 < x[0] < lu[-1] + 0.5 and lL[0] - 0.5 < x[1] < lL[-1] + 0.5):
            continue
        if any(abs(x[0] - y[0]) < 1e-7 and abs(x[1] - y[1]) < 1e-7 for y, _ in out):
            continue
        d = np.linalg.det(_J(x, *args))
        out.append((x, float(d)))
    out.sort(key=lambda t: t[0][0])
    return out
