import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = os.environ.get("CHAINS", ".")

runs = [("SIDM1b halo 1\n(19 kpc core)", f"{S}/chain3_SIDM1b_1.npy", 1.0, "C0"),
        ("SIDM1b halo 4\n(11 kpc core)", f"{S}/chain3_SIDM1b_4.npy", 1.0, "C0"),
        ("SIDM1b halo 2\n(no core)",     f"{S}/chain2_SIDM1b_2.npy", 1.0, "C1"),
        ("CDMb halo 2\n(control)",       f"{S}/chain2_CDMb_2.npy",   0.0, "C2")]

fig, ax = plt.subplots(figsize=(8.5, 5))
for i, (lab, path, truth, col) in enumerate(runs):
    ch = np.load(path); b = ch.shape[0]//2
    sm = 10 ** ch[b:].reshape(-1, ch.shape[-1])[:, 2]
    sm = sm[np.isfinite(sm)]
    lo, mid, hi = np.percentile(sm, [16, 50, 84])
    ax.errorbar(i, mid, yerr=[[mid-lo],[hi-mid]], fmt='o', ms=9, capsize=5,
                color=col, lw=2)
    ax.annotate(f"{mid:.2f}", (i, hi), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=10, color=col)
ax.axhline(1.0, color="C3", lw=2, ls="--", label=r"true $\sigma/m = 1$ (SIDM1b)")
ax.axhline(0.0, color="grey", lw=1.5, ls=":", label=r"true $\sigma/m = 0$ (CDMb)")
ax.set_yscale("symlog", linthresh=0.05)
ax.set_xticks(range(len(runs))); ax.set_xticklabels([r[0] for r in runs], fontsize=9)
ax.set_ylabel(r"recovered $\sigma/m$   [cm$^2$/g]")
ax.set_ylim(-0.02, 4)
ax.set_title("Recovering a known cross-section from EAGLE-50\n"
             "cored halos return ~1.8x the truth; a collisionless halo returns 0.05",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
out = f"{S}/eagle_recovery_summary.png"; fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
