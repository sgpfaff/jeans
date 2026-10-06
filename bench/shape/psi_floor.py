"""Is the flat 2e-4 in r0 a discretisation floor or real linearisation error?

If it is a floor, refining both sides pushes it down and the linearisation
error in r0 is even smaller than measured. If it stays put while the shape
error keeps growing, it is real and happens to be flat.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
warnings.simplefilter("ignore")
from multiprocessing import Pool
GN = 4.302e-6


def one(arg):
    q0, Md, r1, M200, c, a, b, r_grid, n_steps = arg
    import jeans
    from jeans.classes import CDM_profile
    from jeans.fast.solver2d import solve_axisymmetric
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    try:
        o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
        rho1, M1 = o.rho_sph_avg(r1), o.M_encl(r1)
        J = o.compute_potential_moments(r1, L_list=[0, 2], M_list=[0, 0])
        f = solve_axisymmetric(r1, rho1, M1, J_L=J, L_list=(0, 2), Phi_b=pb,
                               n_outer=8, n_steps=n_steps)
        h = jeans.isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=[0, 2],
                             r_grid=r_grid)
        if not f.success or h is None or h.inner is None:
            return None
        rr = 0.999 * r1
        return (r_grid, n_steps, f.max_psi, float(np.max(np.abs(f.phi_L[2]))),
                abs(f.r0 / h.inner.r0 - 1.0),
                abs(f.phi_at(2, rr) / h.inner.phi(2, 0, rr) - 1.0))
    except Exception:
        return None


if __name__ == "__main__":
    # three configurations spanning low, middling and high |phi_2|
    BASE = [(0.90, 3e10, 12.0, 1e12, 10.0, 2.5, 0.4),
            (0.55, 6e10, 12.0, 1e12, 10.0, 2.5, 0.4),
            (0.32, 8e10, 12.0, 1e12, 10.0, 2.5, 0.4)]
    LADDER = [(200, 200), (400, 400), (800, 800), (1600, 1600)]
    args = [b + g for b in BASE for g in LADDER]
    with Pool(12) as p:
        out = [r for r in p.map(one, args, chunksize=1) if r]
    print(f"{'r_grid':>7}{'n_steps':>8}{'max|psi|':>10}{'max|phi2|':>11}"
          f"{'e_r0':>11}{'e_phi2':>11}")
    for row in sorted(out, key=lambda r: (-r[3], r[0])):
        print(f"{row[0]:7d}{row[1]:8d}{row[2]:10.4f}{row[3]:11.4f}"
              f"{row[4]:11.2e}{row[5]:11.2e}")
