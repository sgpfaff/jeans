"""Is the flat ~3% in phi_4 discretisation, under-alternation, or structural?

Three knobs: n_steps/r_grid (discretisation), n_outer (how converged the
Picard alternation is), and nothing else. If refining both leaves it put, what
remains is the linear response's own diagonal approximation, which drops the
2<->4 coupling -- and that would make it structural and not fixable by
turning dials.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
warnings.simplefilter("ignore")
from multiprocessing import Pool
GN = 4.302e-6


def one(arg):
    q0, n, n_outer = arg
    import jeans
    from jeans.classes import CDM_profile
    from jeans.fast.solver2d import solve_axisymmetric
    Md, a, b, r1, M200, c = 6e10, 3.0, 0.28, 12.0, 1e12, 10.0
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    try:
        o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
        rho1, M1 = o.rho_sph_avg(r1), o.M_encl(r1)
        J = o.compute_potential_moments(r1, L_list=[0, 2, 4], M_list=[0, 0, 0])
        f = solve_axisymmetric(r1, rho1, M1, J_L=J, L_list=(0, 2, 4), Phi_b=pb,
                               n_outer=n_outer, n_steps=n)
        h = jeans.isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=[0, 2, 4],
                             r_grid=n)
        if not f.success or h is None or h.inner is None:
            return None
        rr = 0.999 * r1
        return (q0, n, n_outer,
                abs(f.r0 / h.inner.r0 - 1.0),
                abs(f.phi_at(2, rr) / h.inner.phi(2, 0, rr) - 1.0),
                abs(f.phi_at(4, rr) / h.inner.phi(4, 0, rr) - 1.0))
    except Exception:
        return None


if __name__ == "__main__":
    args = [(q0, n, no) for q0 in (0.8, 0.5)
            for n in (200, 400, 800)
            for no in (4, 12)]
    with Pool(12) as p:
        out = [r for r in p.map(one, args, chunksize=1) if r]
    print(f"{'q0':>5}{'n_steps':>9}{'n_outer':>9}{'err r0':>11}"
          f"{'err phi2':>11}{'err phi4':>11}")
    for row in sorted(out):
        print(f"{row[0]:5.2f}{row[1]:9d}{row[2]:9d}{row[3]:11.2e}"
              f"{row[4]:11.2e}{row[5]:11.2e}")
