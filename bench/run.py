"""Run the benchmark suite and write versioned JSON.

Usage:
    PYTHONPATH=src python bench/run.py [section ...]

Sections: cases, cost, gridref, ladder, batch. Default is all of them.
Output goes to bench/results/<section>.json.

Two speed-ups are reported for every case and they mean different things.
`solver_only` compares the interior solve, which is what the reduction
actually replaces. `end_to_end` includes building the outer CDM halo, which
both paths need and neither reduction touches. The package does this
internally, so end_to_end is the number a user would feel and solver_only is
the number that describes the method. Quoting only the larger one would be
misleading.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jeans
from jeans.classes import CDM_profile
from jeans.definitions import GN
from jeans.fast import universal
from jeans.fast.solver import solve_spherical
from jeans.fast.solver2d import solve_axisymmetric

from harness import (Timing, density_error, environment, measure, rel_err,
                     save, speedup_interval)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

MD, A_D, B_D = 6e10, 3.0, 0.28
MSTAR, A_H = 5e10, 4.0


def mn_phi(r, th):
    return -GN * MD / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (A_D + np.sqrt(B_D ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2
    )


def hernquist_phi(r):
    return -GN * MSTAR / (r + A_H)


# name, r1, M200, c, q0, Phi_b, L_list
CASES = [
    ("1D, no baryons",            10, 1e12, 10.0, 1.0, None,          (0,)),
    ("1D + MN disc",              10, 1e12, 10.0, 1.0, mn_phi,        (0,)),
    ("1D + Hernquist spheroid",   30, 1e13,  7.0, 1.0, hernquist_phi, (0,)),
    ("2D L=[0,2], no baryons",    10, 1e12, 10.0, 0.8, None,          (0, 2)),
    ("2D L=[0,2] + disc",         10, 1e12, 10.0, 0.8, mn_phi,        (0, 2)),
    ("2D L=[0,2] + spheroid",     30, 1e13,  7.0, 0.8, hernquist_phi, (0, 2)),
    ("2D L=[0,2,4] + disc",       10, 1e12, 10.0, 0.9, mn_phi,        (0, 2, 4)),
    ("2D L=[0,2,4], no baryons",  20, 5e12,  8.0, 0.75, None,         (0, 2, 4)),
]


def _pkg_call(r1, M200, c, q0, pb, Ls):
    if len(Ls) == 1:
        return lambda: jeans.spherical(r1, M200, c, Phi_b=pb)
    return lambda: jeans.isothermal(r1, M200, c, q0=q0, Phi_b=pb, L_list=list(Ls))


def _outer(r1, M200, c, q0, pb, Ls):
    o = CDM_profile(M200, c, q0=q0, Phi_b=pb)
    return (o.rho_sph_avg(r1), o.M_encl(r1),
            o.compute_potential_moments(r1, L_list=list(Ls),
                                        M_list=[0] * len(Ls)))


def section_cases():
    rows = []
    for name, r1, M200, c, q0, pb, Ls in CASES:
        print(f"  {name} ...", flush=True)

        t_pkg, h = measure(_pkg_call(r1, M200, c, q0, pb, Ls),
                           repeats=5, warmup=1, max_seconds=180.0)
        if h is None:
            print("    package did not converge, skipping")
            continue

        t_outer, bd = measure(lambda: _outer(r1, M200, c, q0, pb, Ls),
                              repeats=5, warmup=1, max_seconds=60.0)
        rho1, M1, JL = bd

        if len(Ls) == 1:
            run = lambda: solve_spherical(r1, rho1, M1, Phi_b=pb)
        else:
            run = lambda: solve_axisymmetric(r1, rho1, M1, J_L=JL,
                                             L_list=Ls, Phi_b=pb)
        t_fast, R = measure(run, repeats=21, warmup=3, max_seconds=60.0)

        end_to_end = Timing(
            median_s=t_fast.median_s + t_outer.median_s,
            min_s=t_fast.min_s + t_outer.min_s,
            q1_s=t_fast.q1_s + t_outer.q1_s,
            q3_s=t_fast.q3_s + t_outer.q3_s,
            n=min(t_fast.n, t_outer.n), warmup=t_fast.warmup)

        acc = {
            "r0_rel": rel_err(R.r0, h.inner.r0),
            "sigma0_rel": rel_err(R.sigma0, h.inner.sigma0),
            "residual": float(R.residual),
        }
        for L in Ls:
            if L == 0:
                continue
            want = h.inner.phi(L, 0, 0.999 * r1)
            acc[f"phi{L}_rel"] = rel_err(R.phi_at(L, 0.999 * r1), want)
            acc[f"phi{L}_pkg"] = float(want)

        rows.append({
            "name": name,
            "params": {"r1": r1, "M200": M200, "c": c, "q0": q0,
                       "baryons": "none" if pb is None else pb.__name__,
                       "L_list": list(Ls)},
            "package": t_pkg, "outer_halo": t_outer,
            "fast_solver": t_fast, "fast_end_to_end": end_to_end,
            "speedup_solver_only": speedup_interval(t_pkg, t_fast),
            "speedup_end_to_end": speedup_interval(t_pkg, end_to_end),
            "accuracy": acc,
        })
        s = rows[-1]["speedup_solver_only"]["median"]
        print(f"    {t_pkg.median_s*1e3:9.1f} ms -> {t_fast.median_s*1e3:7.2f} ms "
              f"({s:,.0f}x, package IQR {t_pkg.iqr_frac:.1%}, "
              f"fast IQR {t_fast.iqr_frac:.1%})", flush=True)
    return {"environment": environment(), "cases": rows}


def section_cost():
    """Accuracy against cost as the step count varies: the real trade-off."""
    r1, M200, c, q0, Ls = 10, 1e12, 10.0, 0.8, (0, 2)
    out = {}
    for label, pb in (("no baryons", None), ("MN disc", mn_phi)):
        rho1, M1, JL = _outer(r1, M200, c, q0, pb, Ls)
        ref = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls, Phi_b=pb,
                                 n_steps=6400)
        rows = []
        for n in (25, 50, 100, 200, 400, 800, 1600):
            run = lambda n=n: solve_axisymmetric(r1, rho1, M1, J_L=JL,
                                                 L_list=Ls, Phi_b=pb, n_steps=n)
            t, R = measure(run, repeats=11, warmup=2, max_seconds=40.0)
            rows.append({"n_steps": n, "time": t,
                         "r0_rel_to_converged": rel_err(R.r0, ref.r0)})
            print(f"  {label} n={n:5d}: {t.median_s*1e3:8.3f} ms  "
                  f"err {rows[-1]['r0_rel_to_converged']:.2e}", flush=True)
        out[label] = rows
    return {"environment": environment(), "series": out}


def section_gridref():
    """The central validation claim, re-measured: does the gap close as N^-2?"""
    out = {}
    for label, (r1, M200, c, q0, pb, Ls) in {
        "1D, no baryons": (10, 1e12, 10.0, 1.0, None, (0,)),
        "1D + MN disc": (10, 1e12, 10.0, 1.0, mn_phi, (0,)),
        "2D L=[0,2] + disc": (10, 1e12, 10.0, 0.8, mn_phi, (0, 2)),
    }.items():
        rho1, M1, JL = _outer(r1, M200, c, q0, pb, Ls)
        if len(Ls) == 1:
            fast = solve_spherical(r1, rho1, M1, Phi_b=pb, n_steps=3200)
        else:
            fast = solve_axisymmetric(r1, rho1, M1, J_L=JL, L_list=Ls,
                                      Phi_b=pb, n_steps=3200)
        rows = []
        for n in (50, 100, 200, 400, 800):
            if len(Ls) == 1:
                h = jeans.spherical(r1, M200, c, Phi_b=pb, r_grid=n)
            else:
                h = jeans.isothermal(r1, M200, c, q0=q0, Phi_b=pb,
                                     L_list=list(Ls), r_grid=n)
            if h is None:
                continue
            rows.append({"r_grid": n, "r0_rel": rel_err(h.inner.r0, fast.r0),
                         "r0_pkg": float(h.inner.r0)})
            print(f"  {label} N={n:4d}: {rows[-1]['r0_rel']:.4e}", flush=True)
        for a, b in zip(rows[:-1], rows[1:]):
            b["order"] = float(np.log(a["r0_rel"] / b["r0_rel"]) / np.log(2.0))
        out[label] = {"fast_r0": float(fast.r0), "rows": rows}
    return {"environment": environment(), "series": out}


def section_ladder():
    """Finite-difference step-size ladder.

    Three series, because the comparison that matters turned out not to be the
    one expected. The package's gradients were reported as losing their plateau
    below s ~ 3e-4 and changing sign; the cause was identified as the spline
    and adaptive-quadrature round-trip in compute_Phi_b_spherical. The D6 fix
    removes that round-trip from the default path, so the two package
    conventions are measured separately here:

      package_spherical  baryon_average="spherical", the original behaviour,
                         which rebuilds Phi_b through compute_Mb (log-log
                         spline) and compute_Phi_b_spherical (adaptive
                         solve_ivp) on every call;
      package_exact      baryon_average="exact", the fixed default, which
                         passes Phi_b straight through;
      fast               the reduced solver on fixed nodes.

    If the diagnosis is right, the first is noisy and the other two are not.
    """
    r1, M200, c, pb = 10, 1e12, 10.0, mn_phi
    steps = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]

    def pkg(avg):
        def f(m200):
            h = jeans.spherical(r1, m200, c, Phi_b=pb, baryon_average=avg)
            return np.nan if h is None else h.inner.r0
        return f

    def fast_r0(m200):
        rho1, M1, _ = _outer(r1, m200, c, 1.0, pb, (0,))
        return solve_spherical(r1, rho1, M1, Phi_b=pb).r0

    out = {"steps": steps}
    for label, f in (("package_spherical", pkg("spherical")),
                     ("package_exact", pkg("exact")),
                     ("fast", fast_r0)):
        vals = []
        for s in steps:
            hi = np.log(f(M200 * (1 + s)))
            lo = np.log(f(M200 * (1 - s)))
            vals.append(float((hi - lo) / (np.log1p(s) - np.log1p(-s))))
            print(f"  {label:18s} s={s:<8g}: {vals[-1]:+.8f}", flush=True)
        out[label] = vals
    return {"environment": environment(), "ladder": out}


def section_batch():
    """Throughput over an ensemble, which is what a sampler actually needs."""
    rng = np.random.default_rng(7)
    draws = []
    while len(draws) < 64:
        r1 = float(rng.uniform(5.0, 25.0))
        M200 = float(10 ** rng.uniform(11.5, 13.0))
        c = float(rng.uniform(6.0, 14.0))
        rho1, M1, _ = _outer(r1, M200, c, 1.0, mn_phi, (0,))
        if universal.matching_ratio_of(r1, rho1, M1) < 0.7:
            draws.append((r1, rho1, M1))

    def run_all():
        return [solve_spherical(r1, rho1, M1, Phi_b=mn_phi)
                for r1, rho1, M1 in draws]

    t, res = measure(run_all, repeats=5, warmup=1, max_seconds=120.0)
    ok = sum(1 for r in res if r.success)
    print(f"  {len(draws)} halos: {t.median_s*1e3:.1f} ms total, "
          f"{t.median_s/len(draws)*1e3:.2f} ms each, {ok}/{len(draws)} converged",
          flush=True)
    return {"environment": environment(),
            "n_halos": len(draws), "total": t,
            "per_halo_ms": t.median_s / len(draws) * 1e3,
            "converged": ok}


SECTIONS = {"cases": section_cases, "cost": section_cost,
            "gridref": section_gridref, "ladder": section_ladder,
            "batch": section_batch}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SECTIONS)
    for key in wanted:
        print(f"[{key}]", flush=True)
        path = save(os.path.join(OUT, f"{key}.json"), SECTIONS[key]())
        print(f"  -> {path}", flush=True)
