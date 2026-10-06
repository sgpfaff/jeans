"""Fisher forecast: which observable constrains sigma/m, and how robustly.

Cheap enough (~1 min) to answer design questions the MCMC takes hours to
answer, so it runs FIRST and the chains follow. Three things it settles:

  * which channel carries the information -- density, isodensity shape,
    circular speed, central dispersion;
  * what survives marginalising the baryons and the outer-halo flattening,
    which is where the literature claim about shape actually lives;
  * what survives an unknown additive offset on a channel, which is exactly
    "use only the gradient" -- a distance error or an axis-ratio calibration
    error is such an offset.

Fisher is local and linear and this posterior is neither: sigma/m has a
ceiling near 1.2 for this halo and the shape observable saturates. Read the
numbers as a design forecast, not a result. Note too that rho and vc are both
derived from the same Psi, so scoring them as independent channels with
independent noise flatters the combined row.
"""
import sys, numpy as np
sys.path.insert(0, "/geir_data/scr/gabrielspace/jeans-fast/bench/information")
import infocontent as IC

T = IC.TRUTH
TH0 = np.array([np.log10(T["M200"]), T["c"], np.log10(T["sigma_m"]),
                np.log10(IC.MD), IC.Q0])
STEP = np.array([0.02, 0.25, 0.02, 0.025, 0.012])
R_VC = np.geomspace(0.5, 12.0, 14)
SIG_VC, SIG_S0 = 0.05, 0.05            # 5% on circular speed and on sigma_0
GA, GW = np.polynomial.legendre.leggauss(8)
TH_G = np.arccos(np.clip(0.5 * (GA + 1.0), -1, 1)); W_G = 0.5 * GW


def vcirc(res, r1):
    import sys, numpy as np
sys.path.insert(0, "/geir_data/scr/gabrielspace/jeans-fast/bench/information")
import infocontent as IC

T = IC.TRUTH
TH0 = np.array([np.log10(T["M200"]), T["c"], np.log10(T["sigma_m"]),
                np.log10(IC.MD), IC.Q0])
STEP = np.array([0.02, 0.25, 0.02, 0.025, 0.012])
R_VC = np.geomspace(0.5, 12.0, 14)
SIG_VC, SIG_S0 = 0.05, 0.05            # 5% on circular speed and on sigma_0
GA, GW = np.polynomial.legendre.leggauss(8)
TH_G = np.arccos(np.clip(0.5 * (GA + 1.0), -1, 1)); W_G = 0.5 * GW


def vcirc(res, r1):
    """sqrt(r <dPhi_tot/dr>) from Psi = -ln(rho/rho0): what a stream feels."""
    out = np.full(R_VC.size, np.nan)
    for i, r in enumerate(R_VC):
        if r >= 0.98 * r1:
            continue
        h = 1e-3 * r
        d = 0.0
        for t, w in zip(TH_G, W_G):
            pp = -np.log(res.rho(r + h, t) / res.rho0)
            pm = -np.log(res.rho(r - h, t) / res.rho0)
            d += w * (pp - pm) / (2 * h)
        g = res.sigma0 ** 2 * d
        out[i] = np.sqrt(r * g) if g > 0 else np.nan
    return out


def fwd(th, a, b):
    lM, c, lsm, lMd, q0 = th
    IC.AD, IC.BD, IC.MD, IC.Q0 = a, b, 10.0 ** lMd, q0
    r1, res = IC.r1_for(10.0 ** lM, c, 10.0 ** lsm)
    if r1 is None:
        return None
    rho, q = IC.observables(res, r1, 10.0 ** lM, c)
    return np.concatenate([np.log(rho), q, np.log(vcirc(res, r1)),
                           [np.log(res.sigma0)]])


NR, NQ, NV = IC.R_RHO.size, IC.R_Q.size, R_VC.size
SIG = np.concatenate([np.full(NR, IC.SIG_RHO), np.full(NQ, IC.SIG_Q),
                      np.full(NV, SIG_VC), [SIG_S0]])
IDX = {"rho": (0, NR), "q": (NR, NR + NQ), "vc": (NR + NQ, NR + NQ + NV),
       "sig0": (NR + NQ + NV, NR + NQ + NV + 1)}


def jac(a, b):
    IC._LAST_R1.clear()
    base = fwd(TH0, a, b)
    if base is None: return None, None
    ok = np.isfinite(base); J = np.zeros((base.size, 5))
    for k in range(5):
        tp, tm = TH0.copy(), TH0.copy()
        tp[k] += STEP[k]; tm[k] -= STEP[k]
        fp, fm = fwd(tp, a, b), fwd(tm, a, b)
        if fp is None or fm is None: return None, None
        J[:, k] = (fp - fm) / (2 * STEP[k])
        ok &= np.isfinite(fp) & np.isfinite(fm)
    return J, ok


def width(J, ok, chans, priors, offs=()):
    sel = np.zeros(ok.size, bool)
    for ch in chans:
        lo, hi = IDX[ch]; sel[lo:hi] = True
    sel &= ok
    if sel.sum() < 3: return np.inf
    Jl = J[sel]
    for ch in offs:                      # free additive offset on that channel
        lo, hi = IDX[ch]
        col = np.zeros(ok.size); col[lo:hi] = 1.0
        Jl = np.column_stack([Jl, col[sel]])
    F = Jl.T @ (Jl / SIG[sel, None] ** 2)
    pr = np.zeros(F.shape[0])
    for k, s in priors.items(): pr[k] = 1.0 / s ** 2
    try: return float(np.sqrt(np.linalg.inv(F + np.diag(pr))[2, 2]))
    except np.linalg.LinAlgError: return np.inf


KNOWN = {3: 1e-6, 4: 1e-6}
CH = [("rho",), ("q",), ("vc",), ("sig0",), ("vc", "sig0"),
      ("rho", "q"), ("rho", "q", "vc", "sig0")]
# Miyamoto-Nagai spans disc -> spheroid -> exact sphere as a -> 0 at fixed
# mass, so one family covers every host shape without changing the potential.
HOSTS = [("disc   a=3.0 b=0.28", 3.0, 0.28), ("E5     a=1.4 b=1.4 ", 1.4, 1.4),
         ("round  a=0.5 b=2.3 ", 0.5, 2.3), ("sphere a=0.0 b=2.8 ", 0.0, 2.8)]
for name, a, b in HOSTS:
    J, ok = jac(a, b)
    print(f"\n=== {name} ===   sigma(log10 sigma/m), dex")
    if J is None: print("  failed"); continue
    print(f"{'channel':26s} {'known':>8s} {'Md.10+Q0':>9s} {'+offsets':>9s}")
    for ch in CH:
        lbl = "+".join(ch)
        w1 = width(J, ok, ch, KNOWN)
        w2 = width(J, ok, ch, {3: 0.10, 4: 0.10})
        w3 = width(J, ok, ch, {3: 0.10, 4: 0.10},
                   offs=tuple(c for c in ch if c in ("q", "vc")))
        print(f"{lbl:26s} {w1:8.3f} {w2:9.3f} {w3:9.3f}")
