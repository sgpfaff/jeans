"""The existence boundary, and whether haloes can reach it.

A halo is not a POINT in the (mu, R) plane: r1 is not observable, so as r1
varies the halo traces a TRAJECTORY. It is reachable by thermalised dark
matter iff that trajectory enters 1/3 < R < R_fold(mu; shape) somewhere.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from jeanie import jaxouter as JO
from jeanie import branch
from jeanie.solver import _Problem

GN = 4.302e-6
d = np.load(os.path.join(HERE, "rfold.npz"))
MU = d["mu"]
LABS = [("thin disc", "#1f77b4"), ("thick disc", "#2ca02c"),
        ("round", "#ff7f0e"), ("compact_point-like", "#d62728")]

fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.9))

# A -- the boundary ----------------------------------------------------------
a = ax[0]
for lab, c in LABS:
    a.plot(MU, d[lab], "-", color=c, lw=2.3, label=lab)
a.axhline(1 / 3, color="k", lw=2.0, ls="--")
a.fill_between(MU, 1 / 3, d["thin disc"], color="#1f77b4", alpha=.10)
a.text(0.08, 0.40, "$R=1/3$: no solution below", fontsize=8.5)
a.set_xscale("log"); a.set_xlabel(r"$\mu = M_d/(4\pi r_1^3\rho_1)$")
a.set_ylabel(r"$R_{\rm fold}$")
a.set_title("A. The existence boundary is a CURVE\n"
            r"all of $(M_{200},c,M_d,r_1)$ enter via $(R,\mu)$ alone",
            fontsize=10.5)
a.legend(frameon=False, fontsize=8.5); a.grid(alpha=.25)

# B -- how much does baryon SHAPE move the boundary? -------------------------
a = ax[1]
ref = d["thin disc"]
for lab, c in LABS:
    a.plot(MU, d[lab] / ref, "-", color=c, lw=2.3, label=lab)
a.axhline(1, color="k", lw=.8)
a.set_xscale("log")
a.set_xlabel(r"$\mu$"); a.set_ylabel(r"$R_{\rm fold}$ / $R_{\rm fold}$(thin disc)")
a.set_title("B. The boundary is a FAMILY indexed by the\n"
            "stellar distribution, which is observable", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5); a.grid(alpha=.25)

# C -- do real haloes get there? ---------------------------------------------
a = ax[2]
for lab, c in LABS[:1]:
    a.plot(MU, d[lab], "-", color="k", lw=2.4, label=r"$R_{\rm fold}$ (thin disc)")
a.axhline(1 / 3, color="k", lw=2.0, ls="--", label="$R=1/3$ floor")
a.fill_between(MU, 1 / 3, d["thin disc"], color="k", alpha=.07)

HAL = [(1e12, 10.0, 6e10, "#1f77b4", r"$10^{12}$, $c{=}10$"),
       (1e12, 16.0, 6e10, "#9467bd", r"$10^{12}$, $c{=}16$"),
       (1e13, 7.0, 3e11, "#2ca02c", r"$10^{13}$, $c{=}7$"),
       (5e11, 12.0, 2e10, "#ff7f0e", r"$5\times10^{11}$, $c{=}12$")]
for M200, c, Md, col, lab in HAL:
    rv = float(JO.einasto_params(M200, c, 0.18)[2])
    r1s = np.geomspace(0.004 * rv, 0.6 * rv, 40)
    Rs, mus = [], []
    for r1 in r1s:
        rho1 = float(JO.rho_sph_avg(r1, M200, c, q0=1.0))
        M1 = float(JO.enclosed_mass(r1, M200, c, q0=1.0))
        den = 4.0 * np.pi * r1 ** 3 * rho1
        Rs.append(M1 / den); mus.append(Md / den)
    Rs, mus = np.array(Rs), np.array(mus)
    a.plot(mus, Rs, "-", color=col, lw=2.0, label=lab)
    covered = mus <= MU.max()
    inside = covered & (Rs > 1 / 3) & (Rs < np.interp(mus, MU, d["thin disc"]))
    a.plot(mus[inside], Rs[inside], "o", color=col, ms=3.5)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$\mu$"); a.set_ylabel("$R$")
a.set_ylim(0.1, 12); a.set_xlim(MU[1], MU.max())
a.set_title("C. Each halo is a TRAJECTORY, not a point\n"
            r"$r_1$ is not observable; dots = inside the allowed band",
            fontsize=10.5)
a.legend(frameon=False, fontsize=8.0, loc="upper left"); a.grid(alpha=.25, which="both")

fig.suptitle("Prong 1: the reachable set of thermalised dark matter. "
             "No cross-section, no halo age, no alternative model fitted.",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = os.path.join(HERE, "boundary.png")
fig.savefig(out, dpi=135); print("wrote", out)
for lab, _ in LABS:
    print(f"{lab:20s} R_fold(0)={d[lab][0]:.4f}  R_fold(mu=6)={d[lab][-1]:.4f}")
