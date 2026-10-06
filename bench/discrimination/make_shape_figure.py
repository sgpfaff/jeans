"""The thing the collisionless module exists for: CDM that responds in SHAPE.

Single pass, not a converged solve: Psi_DM is held at its spherical value and
the flattened baryonic potential is added, so rho = F(Psi_DM(r) + Psi_b(r,th)).
Self-consistency (Psi_DM recomputed from the flattened rho) will amplify this,
not create it. Labelled as such on the figure rather than in a caption nobody
reads.

What this replaces: the package's CDM has rho(r,th) = rho_sph(r_sph(r,th,q0)),
a fixed similar-ellipsoid squashing with the SAME q0 at every radius, and
adiabatic contraction only modifies rho_sph(r) -- a function of radius alone,
which cannot touch the angular structure. So baryons reshaped the radial
profile and left the shape untouched, by construction.
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jeanie.collisionless import einasto_seed, eddington_f, rho_of_psi

HERE = os.path.dirname(os.path.abspath(__file__))
GN = 4.302e-6
RHO2, RM2, ALPHA = 1.0e7, 20.0, 0.18
AD, BD = 3.0, 0.28                      # a thin disc: the strongest test
Q0_STIPULATED = 0.70

r, rho, psi, drho, M = einasto_seed(RHO2, RM2, ALPHA)
E, f = eddington_f(psi, drho)
Em = E[-1]


def psi_b(R, th, Md):
    """-Phi for a Miyamoto-Nagai disc; th measured from the pole."""
    return GN * Md / np.sqrt((R * np.sin(th)) ** 2
                             + (AD + np.sqrt(BD ** 2 + (R * np.cos(th)) ** 2)) ** 2)


def rho_cdm(R, th, Md):
    """One pass: spherical DM potential plus the flattened baryonic one."""
    p = np.interp(R, r, psi) + psi_b(R, th, Md)
    return rho_of_psi(np.atleast_1d(p), E, f)[0], p


def rho_sidm(R, th, Md, sig2):
    """The isothermal counterpart in the SAME potential, for comparison."""
    p = np.interp(R, r, psi) + psi_b(R, th, Md)
    return np.exp((p - Em) / sig2), p


def axis_ratio(R, Md, fn, **kw):
    """q from equal-density points at fixed volume-equivalent radius."""
    def g(q):
        a = fn(R * q ** (-1 / 3.0), np.pi / 2, Md, **kw)[0]
        b = fn(R * q ** (2 / 3.0), 0.0, Md, **kw)[0]
        return np.log(a / b)
    try:
        return brentq(g, 0.15, 3.0)
    except Exception:
        return np.nan


def valid(R, Md):
    """Psi_tot must stay below the seed's Psi_max (panel D of the other fig)."""
    return (np.interp(R, r, psi) + psi_b(R, 0.0, Md)) < Em


RG = np.geomspace(0.3, 60.0, 34)
MDS = [1.0e10, 3.0e10, 6.0e10]
fig, ax = plt.subplots(1, 4, figsize=(19.4, 4.8))
COL = ["#9ecae1", "#4292c6", "#08519c"]

# A -- the derived shape, against the stipulated one -------------------------
a = ax[0]
for Md, c in zip(MDS, COL):
    ok = np.array([valid(R, Md) for R in RG])
    q = np.array([axis_ratio(R, Md, rho_cdm) if o else np.nan
                  for R, o in zip(RG, ok)])
    a.semilogx(RG, q, "-", color=c, lw=2.3,
               label=rf"$M_d={Md/1e10:.0f}\times10^{{10}}M_\odot$")
a.axhline(Q0_STIPULATED, color="#d62728", ls="--", lw=2.2,
          label="what the package assumes\n(constant $q_0$, any $M_d$)")
a.axhline(1.0, color="k", lw=.8)
a.set_xlabel("r [kpc]"); a.set_ylabel("DM axis ratio $q$")
a.set_title("A. CDM shape is now DERIVED\nflattens towards the disc, "
            "recovers sphericity outside", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5, loc="lower right"); a.grid(alpha=.25)

# B -- the density field ------------------------------------------------------
a = ax[1]
Rg = np.geomspace(0.3, 40.0, 80)
tg = np.linspace(0.01, np.pi / 2, 60)
RR, TT = np.meshgrid(Rg, tg, indexing="ij")
Z = np.array([[rho_cdm(RR[i, j], TT[i, j], 6e10)[0]
               for j in range(TT.shape[1])] for i in range(RR.shape[0])])
X, Y = RR * np.sin(TT), RR * np.cos(TT)
lv = np.geomspace(Z.min() * 50, Z.max() * 0.7, 11)
a.contour(X, Y, Z, levels=lv, colors="#08519c", linewidths=1.4)
a.contour(X, -Y, Z, levels=lv, colors="#08519c", linewidths=1.4)
a.axhline(0, color="#d62728", lw=3, alpha=.5)
a.text(21, 0.9, "disc", color="#d62728", fontsize=9)
a.set_xlim(0, 28); a.set_ylim(-20, 20); a.set_aspect("equal")
a.set_xlabel("R [kpc]"); a.set_ylabel("z [kpc]")
a.set_title("B. Isodensity contours, $M_d=6\\times10^{10}$\n"
            "flattened near the disc, round far out", fontsize=10.5)
a.grid(alpha=.2)

# C -- CDM vs SIDM, same potential, both derived ------------------------------
a = ax[2]
Md = 6.0e10
ok = np.array([valid(R, Md) for R in RG])
qc = np.array([axis_ratio(R, Md, rho_cdm) if o else np.nan
               for R, o in zip(RG, ok)])
a.semilogx(RG, qc, "-", color="#08519c", lw=2.6, label="collisionless CDM")
for s2, c in ((0.15 * Em, "#ff7f0e"), (0.30 * Em, "#8c564b")):
    qs = np.array([axis_ratio(R, Md, rho_sidm, sig2=s2) if o else np.nan
                   for R, o in zip(RG, ok)])
    a.semilogx(RG, qs, "--", color=c, lw=2.2,
               label=rf"isothermal, $\sigma_0^2={s2/Em:.2f}\Psi_{{\max}}$")
a.axhline(Q0_STIPULATED, color="#d62728", ls=":", lw=2.0, label="stipulated $q_0$")
a.set_xlabel("r [kpc]"); a.set_ylabel("DM axis ratio $q$")
a.set_title("C. Both sides derived, same baryons\n"
            "this is what makes a shape comparison fair", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5, loc="lower right"); a.grid(alpha=.25)

# D -- how big is the effect we were previously setting to zero? --------------
a = ax[3]
for Md, c in zip(MDS, COL):
    ok = np.array([valid(R, Md) for R in RG])
    q = np.array([axis_ratio(R, Md, rho_cdm) if o else np.nan
                  for R, o in zip(RG, ok)])
    g = np.gradient(q, np.log(RG))
    a.semilogx(RG, g, "-", color=c, lw=2.3,
               label=rf"$M_d={Md/1e10:.0f}\times10^{{10}}$")
a.axhline(0, color="#d62728", ls="--", lw=2.2,
          label="what the package assumes")
a.axhline(0.051, color="green", ls=":", lw=2.0,
          label="SIDM gradient we called a signature")
a.set_xlabel("r [kpc]"); a.set_ylabel(r"$dq/d\ln r$")
a.set_title("D. The gradient CDM was not allowed to have\n"
            "comparable to the SIDM one it was compared against", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5); a.grid(alpha=.25)

fig.suptitle("Collisionless CDM now responds in shape to the baryons "
             "(single pass: $\\Psi_{\\rm DM}$ spherical, $\\Psi_b$ flattened; "
             "self-consistency will amplify, not create, this)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = os.path.join(HERE, "cdm_shape_response.png")
fig.savefig(out, dpi=135)
print("wrote", out)
for Md in MDS:
    ok = np.array([valid(R, Md) for R in RG])
    q = np.array([axis_ratio(R, Md, rho_cdm) if o else np.nan
                  for R, o in zip(RG, ok)])
    g = np.gradient(q, np.log(RG))
    print(f"Md={Md:.1e}: q spans {np.nanmin(q):.3f}-{np.nanmax(q):.3f}, "
          f"max |dq/dlnr| = {np.nanmax(np.abs(g)):.4f}")
