"""Overlay the three posteriors. The 2.35x gain is a degeneracy-breaking
claim, so the figure that shows it is the one where the contours cross."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

TRUTH = [1.0, 10.0, 0.5]      # M200/1e12, c, sigma/m
runs = [(r"density $\bar\rho(r)$", "info_rho.npy",  "C0"),
        (r"shape $q(r)$",          "info_q.npy",    "C1"),
        ("both",                   "info_both.npy", "C2")]
RNG = [(0.35, 2.2), (5.0, 16.0), (0.0, 1.5)]
labels = [r"$M_{200}\ [10^{12}\,M_\odot]$", r"$c$", r"$\sigma/m$ [cm$^2$/g]"]

fig = None
for lab, f, col in runs:
    ch = np.load(f); b = ch.shape[0] // 2
    fl = ch[b:].reshape(-1, 3)
    d = np.column_stack([10 ** fl[:, 0] / 1e12, fl[:, 1], 10 ** fl[:, 2]])
    fig = corner.corner(d, labels=labels, color=col, fig=fig, range=RNG,
                        truths=TRUTH, truth_color="k", bins=35,
                        plot_datapoints=False, plot_density=False,
                        levels=(0.39, 0.86), smooth=1.0,
                        hist_kwargs={"lw": 2, "density": True},
                        contour_kwargs={"linewidths": 2},
                        label_kwargs={"fontsize": 11})

handles = [plt.Line2D([], [], color=c, lw=2.5, label=l) for l, _, c in runs]
handles.append(plt.Line2D([], [], color="k", lw=1.5, ls="-", label="truth"))
fig.axes[1].legend(handles=handles, fontsize=11, loc="center", frameon=False)
fig.suptitle("Where the cross-section information lives\n"
             "model-generated truth; contours are 1 and 2$\\sigma$\n"
             "NOT CONVERGED: $n_{\\rm eff}\\approx10$ per chain, so only the "
             "ordering is meaningful, not the widths",
             fontsize=12, y=1.04)
fig.savefig("corner_information.png", dpi=115, bbox_inches="tight")
print("wrote corner_information.png")
for lab, f, _ in runs:
    ch = np.load(f); b = ch.shape[0]//2; fl = ch[b:].reshape(-1, 3)
    cc = np.corrcoef(fl[:, 1], fl[:, 2])[0, 1]
    print(f"  {lab:22s} corr(c, log sigma/m) = {cc:+.3f}")
