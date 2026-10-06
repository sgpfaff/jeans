"""Measure the shipped gate against path-independent ground truth.

Ground truth is the bracketed monotone inversion, which was checked against
exhaustive enumeration on 38 cases (37 exact agreement to 3e-13; the one
apparent mismatch was the test's own sheet label, not the solver).

Two priors. "wide" is the stream-inference prior and measures the rate that
actually matters operationally. "fold" is conditioned toward deep discs and
large R, where folds are common, and exists only to put enough events in the
sample to measure discrimination with usable error bars.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from jeans.fast.solver import _Problem, solve_spherical, GN
import jeans.fast.branch as B

N_STEPS, N_GL = 200, 16


def nfw_boundary(M200, c, r1, h=0.7, del_c=200.0, Om=0.3, Ol=0.7):
    H0 = h * 100.0 * 1e-3
    rho_crit = 3.0 * H0 ** 2 / (8.0 * np.pi * GN) * (Om + Ol)
    over = del_c / 3.0 * c ** 3 / (np.log1p(c) - c / (1.0 + c))
    rho_s = rho_crit * over
    r_s = np.cbrt(3.0 * M200 / (del_c * 4.0 * np.pi * rho_crit)) / c
    x = r1 / r_s
    return rho_s / (x * (1 + x) ** 2), 4 * np.pi * rho_s * r_s ** 3 * (np.log1p(x) - x / (1 + x))


def mn(Md, a, b):
    return lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)


def draw(rng, kind):
    if kind == "wide":
        M200 = 10 ** rng.uniform(11.0, 13.2); c = rng.uniform(5, 20)
        r1 = rng.uniform(3, 25); Md = 10 ** rng.uniform(9.0, 11.3)
        a = rng.uniform(1.0, 6.0); b = rng.uniform(0.1, 1.0)
        rho1, M1 = nfw_boundary(M200, c, r1)
    else:
        # conditioned on a deep disc and a matching ratio near the boundary
        r1 = rng.uniform(3, 25); a = rng.uniform(0.5, 6.0); b = rng.uniform(0.1, 1.0)
        rho1 = 10 ** rng.uniform(5.5, 7.8)
        R = rng.uniform(0.80, 1.45)
        M1 = R * 4 * np.pi * r1 ** 3 * rho1
        Md = M1 * 10 ** rng.uniform(-1.3, 0.55)
        M200 = c = np.nan
    return dict(r1=r1, a=a, b=b, Md=Md, rho1=float(rho1), M1=float(M1),
                M200=float(M200), c=float(c))


def one(arg):
    kind, seed = arg
    rng = np.random.default_rng(seed)
    d = draw(rng, kind)
    r1, rho1, M1, Md, a, b = d["r1"], d["rho1"], d["M1"], d["Md"], d["a"], d["b"]
    Pb = mn(Md, a, b)
    rec = dict(d)
    rec["R"] = M1 / (4 * np.pi * r1 ** 3 * rho1)
    rec["mu"] = Md / (4 * np.pi * r1 ** 3 * rho1)
    try:
        P = _Problem(r1, rho1, M1, N_STEPS, Pb, N_GL, True)
        t = time.perf_counter(); ref = B.solve_bracketed(P); rec["t_brk"] = time.perf_counter() - t
        rec["brk_ok"] = bool(ref.success); rec["brk_r0"] = float(ref.r0)
        rec["brk_u1"] = float(ref.u1); rec["brk_n"] = int(ref.n_eval)
        rec["brk_reason"] = ref.reason; rec["brk_Rfold"] = float(ref.R_fold)

        t = time.perf_counter()
        s0 = solve_spherical(r1, rho1, M1, Phi_b=Pb, n_steps=N_STEPS, n_gl=N_GL,
                             verify=False)
        rec["t_ramp"] = time.perf_counter() - t
        rec["ramp_ok"] = bool(s0.success); rec["ramp_r0"] = float(s0.r0)
        rec["ramp_res"] = float(s0.residual); rec["ramp_n"] = int(s0.n_residual_evals)
        rec["ramp_reason"] = s0.reason

        t = time.perf_counter()
        s1 = solve_spherical(r1, rho1, M1, Phi_b=Pb, n_steps=N_STEPS, n_gl=N_GL,
                             verify=True)
        rec["t_rampv"] = time.perf_counter() - t
        rec["rampv_ok"] = bool(s1.success); rec["rampv_reason"] = s1.reason

        # plain Newton from the universal seed: no continuation at all
        t = time.perf_counter()
        s2 = solve_spherical(r1, rho1, M1, Phi_b=Pb, n_steps=N_STEPS, n_gl=N_GL,
                             n_ramp=1, verify=False)
        rec["t_n1"] = time.perf_counter() - t
        rec["n1_ok"] = bool(s2.success); rec["n1_r0"] = float(s2.r0)
        rec["n1_n"] = int(s2.n_residual_evals)
        if s2.success and np.isfinite(s2.r0) and s2.r0 > 0:
            Pc2 = _Problem(r1, rho1, M1, N_STEPS, Pb, N_GL, True)
            t = time.perf_counter()
            ok2, why2 = B.certify(Pc2, s2.r0, s2.sigma0)
            rec["t_cert1"] = time.perf_counter() - t
            rec["cert1_ok"] = bool(ok2); rec["cert1_why"] = why2
            rec["cert1_n"] = int(Pc2.n_eval)
        else:
            rec["cert1_ok"] = None; rec["cert1_n"] = 0; rec["t_cert1"] = 0.0

        if s0.success and np.isfinite(s0.r0) and s0.r0 > 0:
            Pc = _Problem(r1, rho1, M1, N_STEPS, Pb, N_GL, True)
            t = time.perf_counter()
            ok, why = B.certify(Pc, s0.r0, s0.sigma0)
            rec["t_cert"] = time.perf_counter() - t
            rec["cert_ok"] = bool(ok); rec["cert_why"] = why
            rec["cert_n"] = int(Pc.n_eval); rec["ramp_u1"] = float(r1 / s0.r0)
        else:
            rec["cert_ok"] = None; rec["cert_n"] = 0; rec["t_cert"] = 0.0
    except Exception as e:
        import traceback
        rec["fatal"] = traceback.format_exc()[-400:]
    return rec


if __name__ == "__main__":
    kind = sys.argv[1]; n = int(sys.argv[2])
    from multiprocessing import Pool
    args = [(kind, 900_000 + i if kind == "wide" else 5_000_000 + i) for i in range(n)]
    t0 = time.time()
    with Pool(60) as pool:
        recs = pool.map(one, args, chunksize=2)
    print(f"{kind}: {n} draws in {time.time()-t0:.0f} s")
    json.dump(recs, open(f"campaign_{kind}.json", "w"))
