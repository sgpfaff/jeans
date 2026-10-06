"""Numba kernels for the reduced isothermal Jeans solvers.

Everything here takes plain arrays and returns plain arrays: no package objects,
no closures over Python state. That is what makes the routines jittable, and it
is also what keeps the step count fixed, which is what makes finite differences
of the result smooth (see the step-size ladder in the write-up).

The monopole system integrated is

    dphi/dr = (r / 3 r0^2) exp(-eta)
    deta/dr = -(3/r) [1 - exp(eta - phi) m0(r)]

with m0(r) = <exp(-dPhi_b(r, theta) / sigma0^2)>_theta the angular average of the
baryon Boltzmann factor. Writing s(r) = -log m0(r) puts the second equation in
the form -(3/r) expm1(eta - phi - s), which is what the kernels below use: the
expm1 matters because eta - phi - s goes to zero at the origin.
"""
import numpy as np
from numba import njit

__all__ = [
    "source_from_grid",
    "rk4_monopole",
    "rk4_monopole_full",
    "rk4_multipole_linear",
]


# --------------------------------------------------------------------------
# baryon source
# --------------------------------------------------------------------------
@njit(cache=True)
def source_from_grid(dphi_b, weights, sig0sq):
    """s(r) = -log < exp(-dPhi_b(r, theta) / sigma0^2) >_theta.

    Parameters
    ----------
    dphi_b : (n_r, n_theta) array
        Phi_b(r_i, theta_j) - Phi_b(0), tabulated once. Independent of sigma0,
        which is what lets the expensive part be cached across Newton iterates.
    weights : (n_theta,) array
        Quadrature weights in cos(theta), normalised to sum to 1.
    sig0sq : float

    Notes
    -----
    Evaluated as a shifted log-sum-exp. The shift is the per-radius minimum of
    dPhi_b/sigma0^2; without it, deep baryon potentials (max phi_b above 10 for a
    Sersic n=8 spheroid) overflow the exponential before the average is taken.
    """
    n_r, n_th = dphi_b.shape
    out = np.empty(n_r)
    for i in range(n_r):
        vmin = dphi_b[i, 0] / sig0sq
        for j in range(1, n_th):
            v = dphi_b[i, j] / sig0sq
            if v < vmin:
                vmin = v
        acc = 0.0
        for j in range(n_th):
            acc += weights[j] * np.exp(-(dphi_b[i, j] / sig0sq - vmin))
        out[i] = vmin - np.log(acc)
    return out


# --------------------------------------------------------------------------
# monopole
# --------------------------------------------------------------------------
@njit(cache=True)
def rk4_monopole(r_nodes, h, r0sq, s_nodes, s_half):
    """Integrate the monopole system from r=0 to r=r_nodes[-1]. Returns (phi1, eta1)."""
    phi = 0.0
    eta = 0.0
    inv3r0sq = 1.0 / (3.0 * r0sq)
    n = r_nodes.shape[0] - 1
    for i in range(n):
        rA = r_nodes[i]
        rM = rA + 0.5 * h
        rB = r_nodes[i + 1]
        sA = s_nodes[i]
        sM = s_half[i]
        sB = s_nodes[i + 1]

        if rA == 0.0:
            k1p = 0.0
            k1e = 0.0
        else:
            k1p = rA * inv3r0sq * np.exp(-eta)
            k1e = -(3.0 / rA) * np.expm1(eta - phi - sA)

        p2 = phi + 0.5 * h * k1p
        e2 = eta + 0.5 * h * k1e
        k2p = rM * inv3r0sq * np.exp(-e2)
        k2e = -(3.0 / rM) * np.expm1(e2 - p2 - sM)

        p3 = phi + 0.5 * h * k2p
        e3 = eta + 0.5 * h * k2e
        k3p = rM * inv3r0sq * np.exp(-e3)
        k3e = -(3.0 / rM) * np.expm1(e3 - p3 - sM)

        p4 = phi + h * k3p
        e4 = eta + h * k3e
        k4p = rB * inv3r0sq * np.exp(-e4)
        k4e = -(3.0 / rB) * np.expm1(e4 - p4 - sB)

        phi += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        eta += h / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)
    return phi, eta


@njit(cache=True)
def rk4_monopole_full(r_nodes, h, r0sq, s_nodes, s_half):
    """As rk4_monopole, but keeping the whole trajectory. Returns (phi[], eta[])."""
    n = r_nodes.shape[0] - 1
    PHI = np.zeros(n + 1)
    ETA = np.zeros(n + 1)
    phi = 0.0
    eta = 0.0
    inv3r0sq = 1.0 / (3.0 * r0sq)
    for i in range(n):
        rA = r_nodes[i]
        rM = rA + 0.5 * h
        rB = r_nodes[i + 1]
        sA = s_nodes[i]
        sM = s_half[i]
        sB = s_nodes[i + 1]

        if rA == 0.0:
            k1p = 0.0
            k1e = 0.0
        else:
            k1p = rA * inv3r0sq * np.exp(-eta)
            k1e = -(3.0 / rA) * np.expm1(eta - phi - sA)

        p2 = phi + 0.5 * h * k1p
        e2 = eta + 0.5 * h * k1e
        k2p = rM * inv3r0sq * np.exp(-e2)
        k2e = -(3.0 / rM) * np.expm1(e2 - p2 - sM)

        p3 = phi + 0.5 * h * k2p
        e3 = eta + 0.5 * h * k2e
        k3p = rM * inv3r0sq * np.exp(-e3)
        k3e = -(3.0 / rM) * np.expm1(e3 - p3 - sM)

        p4 = phi + h * k3p
        e4 = eta + h * k3e
        k4p = rB * inv3r0sq * np.exp(-e4)
        k4e = -(3.0 / rB) * np.expm1(e4 - p4 - sB)

        phi += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        eta += h / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)
        PHI[i + 1] = phi
        ETA[i + 1] = eta
    return PHI, ETA


# --------------------------------------------------------------------------
# multipole, linearised about the spherical background
# --------------------------------------------------------------------------
@njit(cache=True)
def rk4_multipole_linear(r_nodes, h, L, rho_nodes, rho_half, src_nodes, src_half):
    """March one L>0 mode of the linearised multipole system outward from r=0.

    The system is

        dphi_L/dr = mu_L / r^2
        dmu_L/dr  = L(L+1) phi_L + 4 pi G r^2 rho_L

    with rho_L the L-th angular moment of the density. Linearising about the
    spherical background makes rho_L = -rho_0(r) phi_L + src_L(r), so the mode
    is a linear two-point problem: one particular solution plus one homogeneous
    solution, combined to meet the outer boundary condition.

    Returns (phi_p, mu_p, phi_h, mu_h) at the outer node: the particular
    solution (zero initial data, driven by src) and the homogeneous solution
    (regular r^L behaviour), each as a 2-vector at r1.

    Note on why this is linearised rather than shot nonlinearly: the L=4
    boundary functional requires a ~5e4 cancellation against the homogeneous
    r^4 mode, so nonlinear shooting loses about four digits and L=6 does not
    converge at all.
    """
    n = r_nodes.shape[0] - 1
    LL = float(L * (L + 1))

    # particular: phi_L(0) = mu_L(0) = 0
    pp = 0.0
    mp = 0.0
    # homogeneous: regular solution ~ r^L, started from the series at the first
    # non-zero node so that the r^L scaling is exact rather than integrated
    # through the origin.
    r_start = r_nodes[1]
    ph = r_start ** L
    mh = float(L) * r_start ** (L + 1)

    for i in range(n):
        rA = r_nodes[i]
        rM = rA + 0.5 * h
        rB = r_nodes[i + 1]

        for which in range(2):
            if which == 0:
                p = pp
                m = mp
                use_src = 1.0
            else:
                if i == 0:
                    continue
                p = ph
                m = mh
                use_src = 0.0

            # k1
            if rA == 0.0:
                k1p = 0.0
                k1m = 0.0
            else:
                k1p = m / (rA * rA)
                k1m = LL * p - rho_nodes[i] * p * rA * rA + use_src * src_nodes[i]
            p2 = p + 0.5 * h * k1p
            m2 = m + 0.5 * h * k1m
            k2p = m2 / (rM * rM)
            k2m = LL * p2 - rho_half[i] * p2 * rM * rM + use_src * src_half[i]
            p3 = p + 0.5 * h * k2p
            m3 = m + 0.5 * h * k2m
            k3p = m3 / (rM * rM)
            k3m = LL * p3 - rho_half[i] * p3 * rM * rM + use_src * src_half[i]
            p4 = p + h * k3p
            m4 = m + h * k3m
            k4p = m4 / (rB * rB)
            k4m = LL * p4 - rho_nodes[i + 1] * p4 * rB * rB + use_src * src_nodes[i + 1]

            p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
            m += h / 6.0 * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)

            if which == 0:
                pp = p
                mp = m
            else:
                ph = p
                mh = m

    return pp, mp, ph, mh


@njit(cache=True)
def rk4_multipole_trajectory(r_nodes, h, L, rho_nodes, rho_half,
                             src_nodes, src_half, c):
    """phi_L(r) over the whole grid for (particular + c * homogeneous).

    Same march as rk4_multipole_linear, retaining the trajectory. Separate
    rather than merged because the endpoint-only version runs inside the
    root-find and should not pay for the storage.
    """
    n = r_nodes.shape[0] - 1
    out = np.zeros(n + 1)
    LL = float(L * (L + 1))
    pp = 0.0
    mp = 0.0
    r_start = r_nodes[1]
    ph = r_start ** L
    mh = float(L) * r_start ** (L + 1)

    for i in range(n):
        rA = r_nodes[i]
        rM = rA + 0.5 * h
        rB = r_nodes[i + 1]
        for which in range(2):
            if which == 0:
                p = pp
                m = mp
                use = 1.0
            else:
                if i == 0:
                    continue
                p = ph
                m = mh
                use = 0.0
            if rA == 0.0:
                k1p = 0.0
                k1m = 0.0
            else:
                k1p = m / (rA * rA)
                k1m = LL * p - rho_nodes[i] * p * rA * rA + use * src_nodes[i]
            p2 = p + 0.5 * h * k1p
            m2 = m + 0.5 * h * k1m
            k2p = m2 / (rM * rM)
            k2m = LL * p2 - rho_half[i] * p2 * rM * rM + use * src_half[i]
            p3 = p + 0.5 * h * k2p
            m3 = m + 0.5 * h * k2m
            k3p = m3 / (rM * rM)
            k3m = LL * p3 - rho_half[i] * p3 * rM * rM + use * src_half[i]
            p4 = p + h * k3p
            m4 = m + h * k3m
            k4p = m4 / (rB * rB)
            k4m = LL * p4 - rho_nodes[i + 1] * p4 * rB * rB + use * src_nodes[i + 1]
            p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
            m += h / 6.0 * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            if which == 0:
                pp = p
                mp = m
            else:
                ph = p
                mh = m
        out[i + 1] = pp + c * ph
    return out
