"""Linear-response error against the package's nonlinear relaxation, vs max|phi_2|.

solver2d solves the L>0 sector by linear response. The package's relaxation
builds its source as the exact angular integral of exp(-phi_b - sum_L phi_L Z_L)
and Newton-iterates on it, so it makes no linearity assumption and is a valid
independent reference for exactly this question.

q0 is swept well past the realistic range so the curve has a knee rather than
only a plateau.
"""
import os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
warnings.simplefilter("ignore")
from multiprocessing import Pool

GN = 4.302e-6


def one(arg):
    q0, Md, r1, M200, c, a, b = arg
    import jeans
    from jeans.classes import CDM_profile
    from jeanie.solver2d import solve_axisymmetric
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    try:
        o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
        rho1, M1 = o.rho_sph_avg(r1), o.M_encl(r1)
        J = o.compute_potential_moments(r1, L_list=[0, 2], M_list=[0, 0])
        fast = solve_axisymmetric(r1, rho1, M1, J_L=J, L_list=(0, 2), Phi_b=pb,
                                  n_outer=6)
        if not fast.success:
            return None
        t = time.time()
        h = jeans.isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=[0, 2])
        if h is None or h.inner is None:
            return None
        rr = 0.999 * r1
        psi = float(np.max(np.abs(fast.phi_L[2])))
        return dict(q0=q0, Md=Md, r1=r1, M200=M200, c=c, psi=psi,
                    e_r0=abs(fast.r0 / h.inner.r0 - 1.0),
                    e_s0=abs(fast.sigma0 / h.inner.sigma0 - 1.0),
                    e_phi2=abs(fast.phi_at(2, rr) / h.inner.phi(2, 0, rr) - 1.0),
                    t_pkg=time.time() - t)
    except Exception:
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    args = []
    for q0 in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5,
               0.45, 0.4, 0.35, 0.3, 0.25):
        for _ in range(10):
            args.append((q0,
                         10 ** rng.uniform(9.5, 11.0),
                         rng.uniform(5, 25),
                         10 ** rng.uniform(11.6, 12.8),
                         rng.uniform(7, 13),
                         rng.uniform(1.5, 5.0),
                         rng.uniform(0.15, 0.9)))
    t0 = time.time()
    with Pool(28) as p:
        out = [r for r in p.map(one, args, chunksize=1) if r]
    print(f"{len(out)} of {len(args)} converged in {time.time()-t0:.0f} s "
          f"(package median {np.median([r['t_pkg'] for r in out]):.1f} s each)")
    import json
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "psi_err.json"), "w"))
    A = sorted(out, key=lambda r: r["psi"])
    print(f"\n{'|phi_2| band':>16}{'n':>5}{'med e_r0':>11}{'max e_r0':>11}"
          f"{'med e_s0':>11}{'med e_phi2':>12}{'max e_phi2':>12}")
    bands = [(0, .05), (.05, .1), (.1, .15), (.15, .2), (.2, .3),
             (.3, .4), (.4, .5), (.5, .7), (.7, 10)]
    for lo, hi in bands:
        g = [r for r in A if lo <= r["psi"] < hi]
        if not g:
            continue
        f = lambda k: np.array([r[k] for r in g])
        print(f"{f'[{lo:.2f},{hi:.2f})':>16}{len(g):5d}"
              f"{np.median(f('e_r0')):11.2e}{f('e_r0').max():11.2e}"
              f"{np.median(f('e_s0')):11.2e}"
              f"{np.median(f('e_phi2')):12.2e}{f('e_phi2').max():12.2e}")
