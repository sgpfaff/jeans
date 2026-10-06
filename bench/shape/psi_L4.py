"""L=4 under linear response: is the dropped quadratic the leading term?

For L=2 the linearisation drops a correction. For L=4 it may be dropping the
dominant contribution, since phi_2^2 and phi_4 are comparable. If so the L=4
amplitude is not a small-error quantity and should not be presented as one.
"""
import sys, warnings
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
        J = o.compute_potential_moments(r1, L_list=[0, 2, 4], M_list=[0, 0, 0])
        f = solve_axisymmetric(r1, rho1, M1, J_L=J, L_list=(0, 2, 4), Phi_b=pb,
                               n_outer=8)
        h = jeans.isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=[0, 2, 4])
        if not f.success or h is None or h.inner is None:
            return None
        rr = 0.999 * r1
        p2 = float(np.max(np.abs(f.phi_L[2])))
        p4 = float(np.max(np.abs(f.phi_L[4])))
        return (q0, p2, p4, p2 ** 2 / max(p4, 1e-30),
                abs(f.r0 / h.inner.r0 - 1.0),
                abs(f.phi_at(2, rr) / h.inner.phi(2, 0, rr) - 1.0),
                abs(f.phi_at(4, rr) / h.inner.phi(4, 0, rr) - 1.0))
    except Exception:
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    args = []
    for q0 in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        for _ in range(6):
            args.append((q0, 10 ** rng.uniform(9.5, 11.0), rng.uniform(6, 22),
                         10 ** rng.uniform(11.7, 12.7), rng.uniform(7, 13),
                         rng.uniform(1.5, 5.0), rng.uniform(0.15, 0.9)))
    with Pool(24) as p:
        out = [r for r in p.map(one, args, chunksize=1) if r]
    A = np.array(out)
    print(f"{len(A)} of {len(args)} converged with L = (0, 2, 4)\n")
    print(f"{'q0':>5}{'|phi2|':>9}{'|phi4|':>10}{'phi2^2/phi4':>13}"
          f"{'err r0':>10}{'err phi2':>10}{'err phi4':>10}")
    for row in sorted(out, key=lambda r: r[1]):
        print(f"{row[0]:5.2f}{row[1]:9.4f}{row[2]:10.2e}{row[3]:13.2f}"
              f"{row[4]:10.1e}{row[5]:10.1e}{row[6]:10.1e}")
    print(f"\nerr phi_4: median {np.median(A[:,6]):.2e}  p90 {np.percentile(A[:,6],90):.2e}"
          f"  max {A[:,6].max():.2e}")
    print(f"err phi_2: median {np.median(A[:,5]):.2e}")
    print(f"err r0   : median {np.median(A[:,4]):.2e}")
    print(f"phi_2^2 / phi_4: median {np.median(A[:,3]):.2f}  "
          f"(order 1 means the dropped quadratic is the leading term in L=4)")
