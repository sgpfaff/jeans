"""What the collisionless CDM module does, and where it is not ready yet.

Panel C is the validation; panel D is the honest one -- it shows the regime
that is still broken rather than cropping it out.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jeanie.collisionless import einasto_seed, eddington_f, rho_of_psi

HERE = os.path.dirname(os.path.abspath(__file__))
RHO2, RM2, ALPHA = 1.0e7, 20.0, 0.18

r, rho, psi, drho, M = einasto_seed(RHO2, RM2, ALPHA)
E, f = eddington_f(psi, drho)
Em = E[-1]

fig, ax = plt.subplots(2, 3, figsize=(16.2, 9.0))
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"

# A -- the seed ---------------------------------------------------------------
a = ax[0, 0]
m = (r > 0.005 * RM2) & (r < 200 * RM2)
a.loglog(r[m], rho[m], color=BLUE, lw=2.2, label=r"$\rho(r)$  Einasto $\alpha=0.18$")
a.set_xlabel("r [kpc]"); a.set_ylabel(r"$\rho$  [M$_\odot$ kpc$^{-3}$]", color=BLUE)
a.tick_params(axis="y", labelcolor=BLUE)
a2 = a.twinx()
a2.semilogx(r[m], psi[m], color=RED, lw=2.2, ls="--")
a2.set_ylabel(r"$\Psi = -\Phi$  [(km/s)$^2$]", color=RED)
a2.tick_params(axis="y", labelcolor=RED)
a2.axhline(psi.max(), color=GREY, ls=":", lw=1.4)
a2.text(r[m].min() * 1.5, psi.max() * 0.90, r"$\Psi_{\max}$", color=GREY, fontsize=10)
a.set_title("A. The seed halo\nfinite central density, so $\\Psi_{\\max}$ is finite",
            fontsize=10.5)
a.grid(alpha=.25, which="both")

# B -- the distribution function ---------------------------------------------
a = ax[0, 1]
a.loglog(E, f, color=GREEN, lw=2.2)
a.set_xlabel(r"$E$  [(km/s)$^2$]"); a.set_ylabel(r"$f(E)$")
a.set_title(f"B. Isotropic DF by Eddington inversion\n"
            f"positive at all {len(E)} nodes (min {f.min():.1e})", fontsize=10.5)
a.grid(alpha=.25, which="both")

# C -- the validation ---------------------------------------------------------
a = ax[0, 2]
mv = (r > 0.01 * RM2) & (r < 100 * RM2)
back = rho_of_psi(psi[mv], E, f)
err = np.abs(back / rho[mv] - 1.0)
a.loglog(r[mv], err, color=BLUE, lw=1.8)
for y, lab, c in ((np.median(err), f"median {np.median(err):.1e}", RED),
                  (err.max(), f"max {err.max():.1e}", GREY)):
    a.axhline(y, color=c, ls="--", lw=1.4, label=lab)
a.set_xlabel("r [kpc]"); a.set_ylabel(r"$|\rho_{\rm recovered}/\rho_{\rm seed}-1|$")
a.set_title("C. Round trip: re-integrate $f$ in the ORIGINAL potential\n"
            "0.01 to 100 scale radii, the regime the model uses", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25, which="both")

# D -- F(Psi), including the part that is still wrong -------------------------
a = ax[1, 0]
pg = np.geomspace(0.02 * Em, 30 * Em, 400)
Fg = rho_of_psi(pg, E, f)
below, above = pg <= Em, pg > Em
a.loglog(pg[below], Fg[below], color=BLUE, lw=2.2, label=r"$F(\Psi)$, validated")
a.loglog(pg[above], Fg[above], color=RED, lw=2.2, ls="--",
         label=r"$\Psi > \Psi_{\max}$ — NOT ready")
a.axvline(Em, color=GREY, ls=":", lw=1.6)
a.text(Em * 1.15, Fg.min() * 4, r"$\Psi_{\max}$", color=GREY, fontsize=10)
jump = abs(rho_of_psi([Em * 1.001], E, f)[0] / rho_of_psi([Em * 0.999], E, f)[0] - 1)
a.set_xlabel(r"$\Psi$  [(km/s)$^2$]"); a.set_ylabel(r"$F(\Psi)$")
a.set_title(f"D. The density-potential relation\n"
            f"discontinuous by {100*jump:.0f}% at $\\Psi_{{\\max}}$ "
            f"— open issue, recorded as xfail", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25, which="both")

# E -- the contraction response, VALID REGIME ONLY ---------------------------
a = ax[1, 1]
base = rho_of_psi(psi[mv], E, f)
for fac, c in ((1.05, "#9ecae1"), (1.2, BLUE), (1.5, "#08519c")):
    ratio = rho_of_psi(psi[mv] * fac, E, f) / base
    ok = psi[mv] * fac <= Em          # beyond this the open issue contaminates
    a.semilogx(r[mv][ok], ratio[ok], color=c, lw=2.2, label=rf"$\Psi \to {fac}\,\Psi$")
    a.semilogx(r[mv][~ok], ratio[~ok], color=c, lw=1.1, ls=":", alpha=.5)
a.axhline(1, color="k", lw=.8)
rcrit = r[mv][np.argmin(np.abs(psi[mv] * 1.5 - Em))]
a.axvspan(r[mv].min(), rcrit, color=RED, alpha=.07)
a.text(r[mv].min() * 1.3, 1.02, r"$\Psi\times1.5 > \Psi_{\max}$" "\n" "dotted = contaminated",
       color=RED, fontsize=8)
a.set_yscale("log")
a.set_xlabel("r [kpc]"); a.set_ylabel(r"$\rho_{\rm deepened}/\rho$")
a.set_title("E. Contraction is derived, not prescribed\n"
            "solid = valid; dotted = past $\\Psi_{\\max}$, see panel D", fontsize=10.5)
a.legend(frameon=False, fontsize=9, loc="upper right"); a.grid(alpha=.25, which="both")

# F -- the two models in the same framework -----------------------------------
a = ax[1, 2]
x = np.geomspace(0.05, 1.0, 300) * Em
Fc = rho_of_psi(x, E, f)
for s2, c, ls in ((0.25 * Em, "#ff7f0e", "--"), (0.5 * Em, "#8c564b", ":")):
    iso = np.exp(-(Em - x) / s2)
    a.loglog(x / Em, iso / iso[-1], color=c, ls=ls, lw=2.0,
             label=rf"isothermal, $\sigma_0^2={s2/Em:.2f}\,\Psi_{{\max}}$")
a.loglog(x / Em, Fc / Fc[-1], color=BLUE, lw=2.6, label="collisionless $F(\\Psi)$")
a.set_xlabel(r"$\Psi/\Psi_{\max}$"); a.set_ylabel(r"$\rho$, normalised at $\Psi_{\max}$")
a.set_title("F. Same structural role, different function\n"
            r"isothermal is the special case $F=\exp$", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5); a.grid(alpha=.25, which="both")

fig.suptitle("Collisionless CDM with an isotropic DF: "
             r"$\rho=F(\Psi)$, so baryons reshape CDM as they do SIDM",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = os.path.join(HERE, "collisionless.png")
fig.savefig(out, dpi=135)
print("wrote", out)
print(f"round trip median {np.median(err):.2e}  max {err.max():.2e}")
print(f"discontinuity at Psi_max: {100*jump:.1f}%")
