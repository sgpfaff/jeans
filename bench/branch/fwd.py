"""The forward map of the reduced spherical problem with a Miyamoto-Nagai disc.

Claim under test: the problem data enters ONLY through four dimensionless
groups and the solution ONLY through two, and the map

    (u1, Lam) -> (R, mu)      at fixed (alpha1, beta1)

is EXPLICIT -- one ODE integration, no root-find.  If true, the spurious-root
problem is an inversion problem with known geometry, not a robustness problem.

    u1    = r1 / r0                 Lam   = G Md / (sigma0^2 r0)
    alpha1= a / r1                  beta1 = b / r1
    R     = M1 / (4 pi r1^3 rho1)   mu    = Md / (4 pi r1^3 rho1)
"""
import numpy as np
from numba import njit

GN = 4.302e-6


@njit(cache=True)
def _s_of(u, Lam, alpha, beta, xs, ws):
    """s = -log < exp(-dPhi_b/sigma0^2) >, dPhi_b/sigma0^2 = Lam[1/(a+b) - 1/D]."""
    if u == 0.0:
        return 0.0
    n = xs.shape[0]
    inv0 = 1.0 / (alpha + beta)
    vmin = 1e300
    tmp = np.empty(n)
    for j in range(n):
        ct = xs[j]
        st2 = 1.0 - ct * ct
        D = np.sqrt(u * u * st2 + (alpha + np.sqrt(beta * beta + u * u * ct * ct)) ** 2)
        v = Lam * (inv0 - 1.0 / D)
        tmp[j] = v
        if v < vmin:
            vmin = v
    acc = 0.0
    for j in range(n):
        acc += ws[j] * np.exp(-(tmp[j] - vmin))
    return vmin - np.log(acc)


@njit(cache=True)
def integrate(u1, Lam, alpha1, beta1, n_steps, xs, ws):
    """RK4 from u=0 to u1. Returns (phi1, eta1, s1)."""
    alpha = alpha1 * u1
    beta = beta1 * u1
    h = u1 / n_steps
    phi = 0.0
    eta = 0.0
    for i in range(n_steps):
        uA = i * h
        uM = uA + 0.5 * h
        uB = uA + h
        sA = _s_of(uA, Lam, alpha, beta, xs, ws)
        sM = _s_of(uM, Lam, alpha, beta, xs, ws)
        sB = _s_of(uB, Lam, alpha, beta, xs, ws)
        if uA == 0.0:
            k1p = 0.0
            k1e = 0.0
        else:
            k1p = uA / 3.0 * np.exp(-eta)
            k1e = -(3.0 / uA) * np.expm1(eta - phi - sA)
        p2 = phi + 0.5 * h * k1p
        e2 = eta + 0.5 * h * k1e
        k2p = uM / 3.0 * np.exp(-e2)
        k2e = -(3.0 / uM) * np.expm1(e2 - p2 - sM)
        p3 = phi + 0.5 * h * k2p
        e3 = eta + 0.5 * h * k2e
        k3p = uM / 3.0 * np.exp(-e3)
        k3e = -(3.0 / uM) * np.expm1(e3 - p3 - sM)
        p4 = phi + h * k3p
        e4 = eta + h * k3e
        k4p = uB / 3.0 * np.exp(-e4)
        k4e = -(3.0 / uB) * np.expm1(e4 - p4 - sB)
        phi += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        eta += h / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)
    s1 = _s_of(u1, Lam, alpha, beta, xs, ws)
    return phi, eta, s1


def nodes(n=16):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w          # half range in cos(theta)


def forward(u1, Lam, alpha1, beta1, n_steps=200, n_gl=16):
    """(log R, log mu) from (u1, Lam). The explicit direction."""
    xs, ws = nodes(n_gl)
    phi1, eta1, s1 = integrate(float(u1), float(Lam), float(alpha1), float(beta1),
                               int(n_steps), xs, ws)
    logR = phi1 - eta1 + s1 - np.log(3.0)
    logmu = np.log(Lam) - 3.0 * np.log(u1) + phi1 + s1
    return logR, logmu
