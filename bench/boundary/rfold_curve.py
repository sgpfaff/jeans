"""R_fold(mu; shape): the existence boundary with baryons.

The whole of prong 1 reduces to a CURVE, not a grid. All of (M200, c, Md, r1)
enter the criterion 1/3 < R < R_fold only through two dimensionless ratios,

    R  = M1 / (4 pi r1^3 rho1),     mu = Md / (4 pi r1^3 rho1),

plus the baryon SHAPE (a/r1, b/r1). This computes the curve, and checks the
dimensionless claim rather than assuming it.
"""
import sys
import numpy as np

sys.path.insert(0, "/geir_data/scr/gabrielspace/jeans-fast/src")
from jeanie import branch
from jeanie.solver import _Problem

GN = 4.302e-6


def disc(Md, a, b):
    return lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)


def R_fold_at(mu, r1=10.0, rho1=1.0e6, a_over_r1=0.20, b_over_r1=0.03,
              n_steps=200):
    """R_fold for a given mu and baryon shape, at any (r1, rho1)."""
    Md = mu * 4.0 * np.pi * r1 ** 3 * rho1
    P = _Problem(r1, rho1, 1.0, n_steps,
                 disc(Md, a_over_r1 * r1, b_over_r1 * r1), 16, True)
    _, Rf = branch.fold_u1(P)
    return Rf


if __name__ == "__main__":
    print("Is R_fold a function of (mu, shape) alone?  Same mu, different "
          "(r1, rho1):")
    print(f"{'r1':>8s} {'rho1':>10s} {'mu':>6s} {'R_fold':>12s}")
    for r1, rho1 in ((10.0, 1e6), (3.0, 1e7), (40.0, 2.5e5)):
        for mu in (0.0, 0.5):
            print(f"{r1:8.1f} {rho1:10.1e} {mu:6.2f} "
                  f"{R_fold_at(mu, r1, rho1):12.8f}")

    print("\nR_fold(mu) for several baryon shapes "
          "(a/r1, b/r1; thin disc -> round):")
    MU = np.concatenate([[0.0], np.geomspace(0.01, 60.0, 34)])
    SHAPES = [(0.30, 0.03, "thin disc"), (0.20, 0.10, "thick disc"),
              (0.10, 0.10, "round"), (0.02, 0.02, "compact/point-like")]
    out = {}
    for a, b, lab in SHAPES:
        vals = np.array([R_fold_at(m, a_over_r1=a, b_over_r1=b) for m in MU])
        out[lab] = vals
        fin = np.isfinite(vals)
        print(f"  {lab:20s} R_fold: {vals[fin].min():.5f} to "
              f"{vals[fin].max():.5f}   ({fin.sum()}/{len(MU)} found)")
    np.savez("/geir_data/scr/gabrielspace/jeans-fast/bench/boundary/rfold.npz",
             mu=MU, **{k.replace("/", "_"): v for k, v in out.items()})
    print("\nno-baryon limit R_fold(0) =", f"{out['thin disc'][0]:.10f}",
          " vs universal.R_MAX =", f"{__import__('jeanie.universal', fromlist=['x']).R_MAX:.10f}")
