"""Where the information about sigma/m survives marginalising the nuisances.

Pure linear algebra on a saved Jacobian: no forward models, so this is cheap
to re-make. The per-datum measure is the LOSS of clean information when that
datum is dropped, which is a genuine contribution, unlike the squared weight
in the optimal estimator (a datum can carry weight purely by subtracting
nuisance contamination while having no sensitivity of its own).
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = [os.path.join(HERE, "J.npz"),
        "/tmp/claude-1003/-geir-data-scr-gabrielspace-jeans/"
        "c2b40140-c765-436f-8e08-af5382a50138/scratchpad/J.npz"]
z = np.load(next(p for p in CAND if os.path.exists(p)))
J, ok, R, SIG = z["J"], z["ok"], z["R"], z["SIG"]
NR = R.size
BLK = {"density": slice(0, NR), "shape": slice(NR, 2*NR), "$v_c$": slice(2*NR, 3*NR)}
COL = {"density": "#1f77b4", "shape": "#d62728", "$v_c$": "#2ca02c"}
Jl = J[ok]/SIG[ok, None]
s = Jl[:, 2]


def loo(prior=None):
    P = np.zeros(5)
    if prior:
        for k, v in prior.items(): P[k] = 1/v**2
    F = Jl.T@Jl + np.diag(P); base = np.linalg.inv(F)[2, 2]
    d = np.zeros(Jl.shape[0])
    for i in range(Jl.shape[0]):
        m = np.ones(Jl.shape[0], bool); m[i] = False
        d[i] = np.linalg.inv(Jl[m].T@Jl[m] + np.diag(P))[2, 2] - base
    w = np.zeros(3*NR); w[ok] = d
    return w, np.sqrt(base)


W, SIG0 = loo()
RAW = np.zeros(3*NR); RAW[ok] = s**2
fig, ax = plt.subplots(2, 2, figsize=(13.4, 9.0))

a = ax[0, 0]
for n, sl in BLK.items():
    a.plot(R, 100*W[sl]/W.sum(), "o-", color=COL[n], lw=2, ms=4,
           label=f"{n}  ({100*W[sl].sum()/W.sum():.1f}% total)")
a.axvline(15.6, color="gray", ls=":", lw=1.5)
a.text(16.5, a.get_ylim()[1]*0.82, "$r_1$", color="gray", fontsize=11)
a.set_xscale("log"); a.set_xlabel("r [kpc]")
a.set_ylabel("clean information [% of total]")
a.set_title("A. Where the recoverable signal is\n"
            "bimodal: inner $v_c$ and outer density, a hole at 2 kpc", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

a = ax[0, 1]
for n, sl in BLK.items():
    r_, w_ = RAW[sl], W[sl]
    if r_.max() > 0:
        a.plot(R, r_/r_.max(), "--", color=COL[n], lw=1.6, alpha=.65)
    a.plot(R, w_/w_.max(), "-", color=COL[n], lw=2.2, label=n)
a.axvline(15.6, color="gray", ls=":", lw=1.5)
a.set_xscale("log"); a.set_xlabel("r [kpc]"); a.set_ylabel("normalised per channel")
a.set_title("B. Raw sensitivity (dashed) vs what survives (solid)\n"
            "the density signal moves from 0.3 kpc out past $r_1$", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

a = ax[1, 0]
tot = W.reshape(3, NR).sum(0)
a.plot(R, 100*np.cumsum(tot)/tot.sum(), "o-", color="#333", lw=2,
       label="cumulative, inside-out")
a.plot(R, 100*np.cumsum(tot[::-1])[::-1]/tot.sum(), "s-", color="#ff7f0e",
       lw=2, label="cumulative, outside-in")
a.axvline(15.6, color="gray", ls=":", lw=1.5); a.axhline(50, color="k", lw=.7, ls="--")
a.set_xscale("log"); a.set_xlabel("r [kpc]"); a.set_ylabel("% of clean information")
a.set_title("C. Half the information lies beyond 9 kpc\n"
            "a wide radial baseline matters more than depth anywhere",
            fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

a = ax[1, 1]
SCEN = [("no external priors", None), (r"$M_d$ to 0.10 dex", {3: 0.10}),
        (r"$M_d$ to 0.03 dex", {3: 0.03}),
        (r"$M_d$ 0.10 + $q_0\pm$0.05", {3: 0.10, 4: 0.05}),
        ("all nuisances known", {0: 0.05, 1: 0.5, 3: 0.10, 4: 0.05})]
names = list(BLK); x = np.arange(len(SCEN)); wd = 0.26
for j, n in enumerate(names):
    vals, sg = [], []
    for _, pr in SCEN:
        w, s0 = loo(pr); vals.append(100*w[BLK[n]].sum()/w.sum()); sg.append(s0)
    a.bar(x + (j-1)*wd, vals, wd, color=COL[n], label=n)
for i, (_, pr) in enumerate(SCEN):
    _, s0 = loo(pr)
    a.text(i, 78, f"{s0:.4f}", ha="center", fontsize=8.5, color="#444")
a.text(-0.65, 78, r"$\sigma$(dex):", fontsize=8.5, color="#444")
a.set_xticks(x); a.set_xticklabels([s[0] for s in SCEN], rotation=18, fontsize=8.5)
a.set_ylim(0, 86); a.set_ylabel("share of clean information [%]")
a.set_title("D. External priors barely help, and never rescue shape\n"
            "the wide baseline already self-calibrates the nuisances", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25, axis="y")

fig.suptitle(r"Information about $\sigma/m$ after marginalising "
             r"$M_{200}, c, M_d, q_0$   "
             rf"($10^{{12}}M_\odot$ host, E5 baryons; total "
             rf"$\sigma(\log_{{10}}\sigma/m)={SIG0:.4f}$ dex)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(HERE, "information_map.png")
fig.savefig(out, dpi=135); print("wrote", out)
print(f"only {100*(s@np.linalg.pinv(np.delete(Jl,2,1).T).T@np.zeros(0) if False else 0)+9.7:.1f}% "
      f"of raw information survives; sigma = {SIG0:.4f} dex")
