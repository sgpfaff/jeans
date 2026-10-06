"""Is there ANY radius where the DM shape gradient separates SIDM from CDM?

Done in ln q, not q: dm_Q is c/a in a frame fixed to the stellar angular
momentum, so it runs above 1 for haloes prolate about that axis (48% of the
SIDM1 bins). ln q is symmetric under oblate<->prolate inversion; a linear fit
in q is not, and an earlier version of this script got a meaningless answer
that way.

Local gradients, not one global slope: the gradient is strongly
radius-dependent, so a single fitted slope just reports the window chosen.
EAGLE-50 ships the SAME haloes run with CDM, SIDM1 and vdSIDM (their r200
agree to 0.4%), so this is a matched comparison that never goes through the
isothermal model.
"""
import os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/geir_data/scr/gabrielspace/jeans/examples/data/EAGLE-50"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("CDMb", "CDM", "#1f77b4"), ("SIDM1b", r"SIDM $\sigma/m{=}1$", "#d62728"),
        ("vdSIDMb", "vdSIDM", "#ff7f0e")]
NMIN = 10_000                      # particles per shell; below this q is noise
XG = np.geomspace(3e-3, 1.0, 22)   # r/r200
XC = np.sqrt(XG[1:]*XG[:-1])


def profiles(key):
    with h5py.File(f"{DATA}/{key}_axisymmetric_shape_profiles.hdf5", "r") as f:
        Q, N, RE, r200 = f["dm_Q"][:], f["dm_N"][:], f["reff"][:], f["r200"][:]
    lnq, grad = [], []
    for i in range(Q.shape[0]):
        x, q, n = RE[i]/r200[i], Q[i], N[i]
        m = np.isfinite(q) & (q > 0) & (n > NMIN) & np.isfinite(x) & (x > 0)
        if m.sum() < 6: continue
        lq = np.interp(XG, x[m], np.log(q[m]), left=np.nan, right=np.nan)
        lq[(XG < x[m].min()) | (XG > x[m].max())] = np.nan
        lnq.append(lq)
        grad.append(np.diff(lq)/np.diff(np.log(XG)))     # local d ln q / d ln r
    return np.array(lnq), np.array(grad)


P = {k: profiles(k) for k, _, _ in RUNS}
fig, ax = plt.subplots(1, 3, figsize=(15.4, 4.9))

for key, lab, col in RUNS:
    lnq = P[key][0]
    mu = np.nanmean(lnq, 0); sd = np.nanstd(lnq, 0)
    ax[0].plot(XG, mu, "-", color=col, lw=2.2, label=lab)
    ax[0].fill_between(XG, mu-sd, mu+sd, color=col, alpha=.18)
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$r/r_{200}$"); ax[0].set_ylabel(r"$\ln q$")
ax[0].set_title(f"A. Shape profiles, mean $\\pm$ halo-to-halo scatter\n"
                f"5 matched haloes/run, shells with $N>${NMIN:,}", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25)

for key, lab, col in RUNS:
    g = P[key][1]
    mu = np.nanmean(g, 0); sd = np.nanstd(g, 0)
    ax[1].plot(XC, mu, "-", color=col, lw=2.2, label=lab)
    ax[1].fill_between(XC, mu-sd, mu+sd, color=col, alpha=.18)
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xscale("log"); ax[1].set_xlabel(r"$r/r_{200}$")
ax[1].set_ylabel(r"local $d\ln q/d\ln r$")
ax[1].set_title("B. The gradient is strongly radius-dependent\n"
                "one fitted slope just reports the window you chose", fontsize=10.5)
ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25)

gc, gs, gv = P["CDMb"][1], P["SIDM1b"][1], P["vdSIDMb"][1]
for g, lab, col in ((gs, r"SIDM1 $-$ CDM", "#d62728"), (gv, "vdSIDM $-$ CDM", "#ff7f0e")):
    sep = np.abs(np.nanmean(g, 0) - np.nanmean(gc, 0)) / np.sqrt(
        np.nanstd(gc, 0)**2 + np.nanstd(g, 0)**2)
    ax[2].plot(XC, sep, "o-", color=col, lw=2, ms=4, label=lab)
    k = np.nanargmax(np.where(np.isfinite(sep), sep, -1))
    print(f"{lab:16s} best separation {sep[k]:.2f} sigma at r/r200 = {XC[k]:.3f}")
ax[2].axhline(1, color="k", ls="--", lw=1)
ax[2].axhline(3, color="green", ls=":", lw=1.5)
ax[2].text(3.4e-3, 3.1, "3$\\sigma$", color="green", fontsize=9)
ax[2].set_xscale("log"); ax[2].set_xlabel(r"$r/r_{200}$")
ax[2].set_ylabel(r"separation / halo-to-halo scatter  [$\sigma$]")
ax[2].set_title("C. Is there ANY radius that works?\n"
                "separation in units of the CDM scatter", fontsize=10.5)
ax[2].legend(frameon=False, fontsize=9); ax[2].grid(alpha=.25)

fig.suptitle("EAGLE-50: does the DM shape gradient separate SIDM from CDM, "
             "and at what radius?  ($M_{200}=10^{13.3}-10^{14.2}M_\\odot$, "
             "clusters not galaxies)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(HERE, "eagle_gradient.png")
fig.savefig(out, dpi=135); print("wrote", out)
