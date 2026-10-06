import os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
warnings.simplefilter("ignore")
from jeans.definitions import GN
from jeans.fast import outer
from jeans.fast.solver2d import solve_axisymmetric
from jeans.classes import CDM_profile
from multiprocessing import Pool


def one(seed):
    rng = np.random.default_rng(seed)
    M200 = 10 ** rng.uniform(11.5, 13.0); c = rng.uniform(6, 14)
    Md = 10 ** rng.uniform(9.5, 11.0); a = rng.uniform(1.5, 5.0)
    b = rng.uniform(0.15, 0.9); r1 = rng.uniform(4, 30); q0 = rng.uniform(0.4, 0.95)
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    try:
        o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
        rho1, M1 = o.rho_sph_avg(r1), o.M_encl(r1)
        J = o.compute_potential_moments(r1, L_list=[0, 2], M_list=[0, 0])
        res = solve_axisymmetric(r1, rho1, M1, J_L=J, L_list=(0, 2), Phi_b=pb)
        if not res.success:
            return None
        return (q0, float(np.max(np.abs(res.phi_L[2]))), r1, Md / M1)
    except Exception:
        return None


if __name__ == "__main__":
    with Pool(24) as p:
        out = [r for r in p.map(one, range(8000, 8240), chunksize=1) if r]
    A = np.array(out)
    print(f"{len(A)} converged configurations, q0 in [0.4, 0.95]")
    print(f"  max|phi_2|: median {np.median(A[:,1]):.4f}  p90 {np.percentile(A[:,1],90):.4f}"
          f"  max {A[:,1].max():.4f}")
    print(f"  documented validated range in the solver2d docstring: 0.272")
    over = (A[:, 1] > 0.272).mean() * 100
    print(f"  fraction exceeding it: {over:.1f}%")
    for lo, hi in ((0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 0.95)):
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum():
            print(f"  q0 in [{lo:.2f},{hi:.2f}): n={m.sum():3d}  "
                  f"median {np.median(A[m,1]):.4f}  max {A[m,1].max():.4f}")
