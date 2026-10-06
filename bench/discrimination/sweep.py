"""Is the optimal statistic universal, or does it depend where you stand?

At each point in (M200, c, sigma/m, Md, q0) compute the score-compression
weights w(theta) -- the optimal nuisance-marginalised estimator of
log10(sigma/m). Stack them and take the SVD. One dominant singular value means
a single compression works everywhere and the leading left singular vector IS
it; a flat spectrum means the best statistic is parameter-dependent and only
trends can be reported.
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "information"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import infocontent as IC

R = np.geomspace(0.3, 40.0, 24); NR = R.size
IC.R_RHO = R.copy(); IC.R_Q = R.copy()
SIG = np.concatenate([np.full(NR, 0.05), np.full(NR, 0.02), np.full(NR, 0.05)])
BLK = dict(density=slice(0, NR), shape=slice(NR, 2*NR), vc=slice(2*NR, 3*NR))
GA, GW = np.polynomial.legendre.leggauss(8)
TH_G = np.arccos(np.clip(0.5*(GA+1.0), -1, 1)); W_G = 0.5*GW
STEP = np.array([0.02, 0.25, 0.02, 0.025, 0.012])


def fwd(th, ad, bd):
    lM, c, lsm, lMd, q0 = th
    IC.AD, IC.BD, IC.MD, IC.Q0 = ad, bd, 10.0**lMd, q0
    IC._LAST_R1.clear()
    r1, res = IC.r1_for(10.0**lM, c, 10.0**lsm)
    if r1 is None: return None
    rho, q = IC.observables(res, r1, 10.0**lM, c)
    vc = np.full(NR, np.nan)
    for i, r in enumerate(R):
        if r >= 0.98*r1: continue
        h = 1e-3*r; d = 0.0
        for t, w in zip(TH_G, W_G):
            d += w*(-np.log(res.rho(r+h, t)/res.rho0)
                    + np.log(res.rho(r-h, t)/res.rho0))/(2*h)
        g = res.sigma0**2*d
        if g > 0: vc[i] = np.sqrt(r*g)
    return np.concatenate([np.log(rho), q, np.log(vc)])


def weights(th, ad, bd):
    base = fwd(th, ad, bd)
    if base is None: return None
    ok = np.isfinite(base); J = np.zeros((base.size, 5))
    for k in range(5):
        tp, tm = th.copy(), th.copy(); tp[k] += STEP[k]; tm[k] -= STEP[k]
        fp, fm = fwd(tp, ad, bd), fwd(tm, ad, bd)
        if fp is None or fm is None: return None
        J[:, k] = (fp - fm)/(2*STEP[k]); ok &= np.isfinite(fp) & np.isfinite(fm)
    Jl = J[ok]/SIG[ok, None]
    F = Jl.T @ Jl
    if np.linalg.cond(F) > 1e12: return None
    Fi = np.linalg.inv(F)
    w = np.zeros(3*NR); w[ok] = Jl @ Fi[:, 2]
    return w, float(np.sqrt(Fi[2, 2])), th


rng = np.random.default_rng(11)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 44
rows = []
for n in range(N):
    th = np.array([rng.uniform(11.5, 13.0), rng.uniform(6.0, 16.0),
                   rng.uniform(np.log10(0.08), np.log10(1.0)),
                   rng.uniform(9.7, 11.2), rng.uniform(0.55, 0.90)])
    ad = rng.uniform(0.4, 3.0); bd = rng.uniform(0.3, 2.4)
    out = weights(th, ad, bd)
    if out is None:
        print(f"[{n:3d}] no solution", flush=True); continue
    w, s, th = out
    rows.append((w, s, th, ad, bd))
    print(f"[{n:3d}] logM={th[0]:5.2f} c={th[1]:5.2f} logSm={th[2]:+5.2f} "
          f"logMd={th[3]:5.2f} q0={th[4]:4.2f}  sigma={s:.4f} dex", flush=True)

W = np.array([r[0] for r in rows])
np.savez(os.path.join(HERE, "sweep.npz"), W=W, R=R,
         sig=np.array([r[1] for r in rows]),
         th=np.array([r[2] for r in rows]),
         disc=np.array([[r[3], r[4]] for r in rows]))
Wn = W/np.linalg.norm(W, axis=1, keepdims=True)
U, S, Vt = np.linalg.svd(Wn, full_matrices=False)
print(f"\n{len(rows)} usable points.  SVD of the unit-normalised weight vectors:")
print("  singular values:", np.round(S[:6], 3))
print(f"  variance explained by mode 1: {100*S[0]**2/np.sum(S**2):.1f}%"
      f"   by modes 1-2: {100*np.sum(S[:2]**2)/np.sum(S**2):.1f}%")
v1 = Vt[0]
print(f"  mode-1 channel split: " + "  ".join(
    f"{k} {100*np.sum(v1[s]**2)/np.sum(v1**2):.1f}%" for k, s in BLK.items()))
L = np.log(R)
for k, s in BLK.items():
    vi = v1[s]; m = np.abs(vi) > 0
    if m.sum() < 4: continue
    B = np.vstack([np.ones(m.sum()), L[m], L[m]**2]).T
    Q, _ = np.linalg.qr(B); cc = (Q.T @ vi[m])**2/(vi[m] @ vi[m])
    print(f"    {k:8s} const {100*cc[0]:5.1f}%  grad {100*cc[1]:5.1f}%  "
          f"curv {100*cc[2]:5.1f}%  higher {100*(1-cc.sum()):5.1f}%")
print(f"\n  alignment |cos| of each point with mode 1: "
      f"median {np.median(np.abs(Wn@v1)):.3f}, "
      f"10th pct {np.percentile(np.abs(Wn@v1), 10):.3f}")
