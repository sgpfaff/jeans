import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TRUTH = 0.5
runs = [("density  " + r"$\bar\rho(r)$" + "\n(X-Stream-like)", "info_rho.npy", "C0"),
        ("shape  " + r"$q(r)$" + "\n(Curve-Away-like)", "info_q.npy", "C1"),
        ("both", "info_both.npy", "C2")]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4),
                        gridspec_kw={"width_ratios": [1.5, 1]})
for lab, f, col in runs:
    ch = np.load(f); b = ch.shape[0] // 2
    sm = 10 ** ch[b:].reshape(-1, 3)[:, 2]
    axs[0].hist(sm, bins=60, range=(0, 1.6), histtype="step", lw=2,
                density=True, color=col, label=lab.replace("\n", " "))
    lo, mid, hi = np.percentile(sm, [16, 50, 84])
    i = [r[1] for r in runs].index(f)
    axs[1].errorbar(i, mid, yerr=[[mid-lo],[hi-mid]], fmt="o", ms=10,
                    capsize=6, lw=2.5, color=col)
    axs[1].annotate(f"width\n{(hi-lo)/mid:.2f}", (i, hi), fontsize=9,
                    ha="center", textcoords="offset points", xytext=(0, 12),
                    color=col)
for a in axs:
    a.axhline if a is axs[1] else None
axs[0].axvline(TRUTH, color="k", ls="--", lw=2, label="truth")
axs[0].set_xlabel(r"$\sigma/m$  [cm$^2$/g]"); axs[0].set_ylabel("posterior density")
axs[0].legend(fontsize=9); axs[0].set_title("posteriors")
axs[1].axhline(TRUTH, color="k", ls="--", lw=2)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels([r[0] for r in runs], fontsize=9)
axs[1].set_ylabel(r"$\sigma/m$  [cm$^2$/g]"); axs[1].set_ylim(0, 1.3)
axs[1].set_title("fractional width of the constraint"); axs[1].grid(alpha=.3, axis="y")
fig.suptitle("Where is the information about the cross-section?\n"
             "model-generated truth, so there is no misspecification — "
             "density and shape carry equal information, and combining them is "
             "worth 2.35x, not $\\sqrt{2}$", fontsize=11)
plt.tight_layout()
fig.savefig("information_content.png", dpi=120, bbox_inches="tight")
print("wrote information_content.png")
for lab, f, _ in runs:
    ch = np.load(f); b = ch.shape[0]//2; sm = 10**ch[b:].reshape(-1,3)[:,2]
    lo, mid, hi = np.percentile(sm, [16,50,84])
    print(f"  {lab.splitlines()[0]:12s} {mid:.3f} +{hi-mid:.3f} -{mid-lo:.3f}"
          f"   width {(hi-lo)/mid:.3f}")
