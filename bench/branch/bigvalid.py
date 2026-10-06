"""Hard validation of the bracketed solver against exhaustive enumeration.

Enumeration cannot miss a root and cannot follow a path, so it is the only
reference that does not share the failure mode. The sheet-1 root is identified
as the one whose det > 0 AND which is connected to u1 -> 0 by a det > 0 segment
-- not by the u1 < 22.544 shortcut, which is what the earlier check used and
which is only sufficient, not necessary.
"""
import os
import sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enumerate_roots as E, fwd
from jeanie.solver import _Problem, GN
import jeanie.branch as B
from multiprocessing import Pool

xs, ws = fwd.nodes(16)


def det_red(lu, lL, a1, b1, n=200, e=1e-5):
    o = np.zeros((2, 2))
    for k in range(2):
        du, dL = (e, 0.0) if k == 0 else (0.0, e)
        for sgn in (1, -1):
            p, et, s = fwd.integrate(np.exp(lu + sgn * du), np.exp(lL + sgn * dL),
                                     a1, b1, n, xs, ws)
            o[0, k] += sgn * (p - et + s) / (2 * e)
            o[1, k] += sgn * (p + s) / (2 * e)
    o[1, 0] += -3.0; o[1, 1] += 1.0
    return o[0, 0] * o[1, 1] - o[0, 1] * o[1, 0]


def sheet1_of(roots, a1, b1):
    """Walk each root down to u1 -> 0 at fixed Lam; keep the one that stays det>0."""
    out = []
    for x, d in roots:
        if d <= 0:
            continue
        good = True
        for lu in np.linspace(x[0], np.log(0.02), 32)[1:]:
            if det_red(lu, x[1], a1, b1) <= 0:
                good = False
                break
        if good:
            out.append(x)
    return out


def one(seed):
    rng = np.random.default_rng(seed)
    kind = "wide" if seed % 2 else "fold"
    r1 = rng.uniform(3, 25); a = rng.uniform(0.5, 6.0); b = rng.uniform(0.1, 1.0)
    rho1 = 10 ** rng.uniform(5.5, 7.8)
    R = rng.uniform(0.35, 1.25) if kind == "wide" else rng.uniform(0.80, 1.45)
    M1 = R * 4 * np.pi * r1 ** 3 * rho1
    Md = M1 * 10 ** rng.uniform(-1.3, 0.55)
    mu = Md / (4 * np.pi * r1 ** 3 * rho1)
    a1, b1 = a / r1, b / r1
    try:
        roots = E.all_roots(np.log(R), np.log(mu), a1, b1)
        s1 = sheet1_of(roots, a1, b1)
        pb = (lambda Md, a, b: lambda rr, th: -GN * Md / np.sqrt(
            rr**2*np.sin(th)**2 + (a + np.sqrt(b**2 + rr**2*np.cos(th)**2))**2))(Md, a, b)
        P = _Problem(r1, rho1, M1, 200, pb, 16, True)
        br = B.solve_bracketed(P)
        return dict(R=R, mu=mu, n_roots=len(roots), n_sheet1=len(s1),
                    u_enum=float(np.exp(s1[0][0])) if s1 else np.nan,
                    u_brk=float(br.u1) if br.success else np.nan,
                    ok=bool(br.success), reason=br.reason)
    except Exception as e:
        return dict(fatal=repr(e)[:120])


if __name__ == "__main__":
    t0 = time.time()
    with Pool(48) as pool:
        recs = pool.map(one, range(7000, 7300), chunksize=1)
    good = [r for r in recs if "fatal" not in r]
    print(f"{len(good)} cases in {time.time()-t0:.0f} s  "
          f"({len(recs)-len(good)} fatal)")
    multi = sum(1 for r in good if r["n_sheet1"] > 1)
    print(f"targets with >1 sheet-1 root (injectivity violations): {multi}")
    agree = dis = both_none = miss = phantom = 0
    worst = 0.0
    for r in good:
        he, hb = np.isfinite(r["u_enum"]), np.isfinite(r["u_brk"])
        if not he and not hb: both_none += 1
        elif he and not hb: miss += 1
        elif hb and not he: phantom += 1
        else:
            rel = abs(r["u_brk"] / r["u_enum"] - 1)
            worst = max(worst, rel)
            if rel < 1e-6: agree += 1
            else: dis += 1
    print(f"  agree           {agree}")
    print(f"  DISAGREE        {dis}")
    print(f"  both: no root   {both_none}")
    print(f"  bracket MISSED  {miss}")
    print(f"  bracket PHANTOM {phantom}")
    print(f"  worst relative difference where both found a root: {worst:.3e}")
    for r in good:
        he, hb = np.isfinite(r["u_enum"]), np.isfinite(r["u_brk"])
        if (he != hb) or (he and hb and abs(r["u_brk"]/r["u_enum"]-1) > 1e-6):
            print(f"    R={r['R']:.4f} mu={r['mu']:.4f} nroots={r['n_roots']} "
                  f"nsheet1={r['n_sheet1']} u_enum={r['u_enum']:.4f} "
                  f"u_brk={r['u_brk']:.4f} [{r['reason']}]")
