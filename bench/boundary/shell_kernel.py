"""Shell kernels: how a thin shell of baryons at radius s moves the boundary.

To first order in baryon amplitude both the fold LOCATION and the fold VALUE
are linear functionals of the baryon mass distribution,

    delta u1_fold = int K_u(s) dm(s),   delta R_fold = int K_R(s) dm(s),

so one pass over shells gives the boundary for ANY baryon distribution without
re-solving per shape. That matters because R_fold moves by a factor 2.8 across
plausible stellar distributions.

Two different kernels, and they are easy to conflate: branch.py documents K_u,
positive inside 0.425 r1 and negative outside it. Prong 1's criterion is on R,
so it needs K_R. This computes both, which also re-tests the documented 0.425
crossing.

Masses are in units of 4 pi r1^3 rho1, so the kernels are dimensionless and,
like R_fold itself, depend only on s/r1.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from jeanie import branch
from jeanie.solver import _Problem

GN = 4.302e-6
R1, RHO1 = 10.0, 1.0e6
UNIT = 4.0 * np.pi * R1 ** 3 * RHO1          # mass unit


def shell(mu_s, s_over_r1):
    """Thin spherical shell: Phi = -Gm/s inside, -Gm/r outside."""
    m, s = mu_s * UNIT, s_over_r1 * R1
    return lambda r, th: -GN * m / np.maximum(np.asarray(r, float), s)


def fold(phi_b, n_steps=400):
    P = _Problem(R1, RHO1, 1.0, n_steps, phi_b, 16, True)
    # NOT the default lo=U_SAFE: branch.py records that a shell at 0.6 r1 moves
    # the first fold inward to 22.37768, below U_SAFE, and the scan then skips
    # it and silently returns the SECOND fold at ~244. Measured here as K_u
    # jumping to 2.2e4 with a 50% linearity error for every s >= 0.45 r1.
    return branch.fold_u1(P, lo=20.0)


if __name__ == "__main__":
    u0, R0 = fold(lambda r, th: np.zeros_like(np.asarray(r, float)))
    print(f"no-baryon fold: u1 = {u0:.6f}, R_fold = {R0:.8f}\n")

    S = np.concatenate([np.linspace(0.05, 1.0, 20), [1.1, 1.3, 1.6]])
    AMP = (0.01, 0.02)                       # two amplitudes -> linearity check
    print(f"{'s/r1':>7s} {'K_u':>11s} {'K_R':>11s} {'lin(K_u)':>9s} {'lin(K_R)':>9s}")
    Ku, KR = [], []
    for s in S:
        est_u, est_R = [], []
        for a in AMP:
            u, R = fold(shell(a, s))
            est_u.append((u - u0) / a)
            est_R.append((R - R0) / a)
        Ku.append(est_u[0]); KR.append(est_R[0])
        print(f"{s:7.3f} {est_u[0]:11.4f} {est_R[0]:11.5f} "
              f"{abs(est_u[1]/est_u[0]-1) if est_u[0] else 0:9.2e} "
              f"{abs(est_R[1]/est_R[0]-1) if est_R[0] else 0:9.2e}")
    Ku, KR = np.array(Ku), np.array(KR)
    np.savez(os.path.join(HERE, "kernels.npz"), s=S, Ku=Ku, KR=KR, u0=u0, R0=R0)

    inside = S < 1.0
    sgn = np.sign(Ku[inside])
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross):
        i = cross[0]
        z = S[i] + (S[i+1]-S[i]) * (-Ku[i]) / (Ku[i+1]-Ku[i])
        print(f"\nK_u sign change at s/r1 = {z:.4f}   (branch.py documents 0.425)")
    print(f"K_R sign: {'all negative' if np.all(KR[inside] < 0) else 'MIXED'} inside r1")
    out = S > 1.0
    print(f"beyond r1 (should vanish): max |K_u| = {np.abs(Ku[out]).max():.3e}, "
          f"max |K_R| = {np.abs(KR[out]).max():.3e}")
