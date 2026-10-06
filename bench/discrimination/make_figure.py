"""Six panels on where SIDM differs from CDM, and where sigma/m is measurable.

The two are not the same question and the figure is built to show that. The
estimation panels ask how the observables move with sigma/m; the discrimination
panels ask what residual survives after CDM refits every nuisance it has.
"""
import sys, os, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "information"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import infocontent as IC
from jeanie import jaxouter as JO

GN = 4.302e-6
R = np.geomspace(0.3, 40.0, 24); Rj = jnp.asarray(R); NR = R.size
IC.R_RHO = R.copy(); IC.R_Q = R.copy()
IC.AD = AD = 1.4; IC.BD = BD = 1.4
SR, SQ, SV = 0.05, 0.02, 0.05
GA, GW = np.polynomial.legendre.leggauss(8)
TH_G = np.arccos(np.clip(0.5*(GA+1.0), -1, 1)); W_G = 0.5*GW


def _obs(p, ac):
    M200, c, Md, q0 = p
    phib = lambda r, th: -GN*Md/jnp.sqrt(
        r**2*jnp.sin(th)**2 + (AD + jnp.sqrt(BD**2 + r**2*jnp.cos(th)**2))**2)
    kw = dict(halo_type="NFW")   # match IC.observables, which takes halo_profile defaults
    if ac: kw.update(AC_prescription="Cautun", Phi_b=phib)
    f = JO.halo_profile(M200, c, **kw)
    rho = jax.vmap(lambda r: JO.rho_sph_avg(r, M200, c, q0=q0, rho_sph=f))(Rj)
    Mdm = jax.vmap(lambda r: JO.enclosed_mass(r, M200, c, q0=q0, rho_sph=f))(Rj)
    Mb = jax.vmap(lambda r: JO.mb_spherical(phib, r))(Rj)
    return jnp.log(rho), jnp.full(NR, q0), jnp.log(jnp.sqrt(GN*(Mdm+Mb)/Rj))


CDM = jax.jit(lambda p: _obs(p, True))


def sidm(sm):
    IC.MD, IC.Q0 = 6e10, 0.7
    IC._LAST_R1.clear()
    r1, res = IC.r1_for(1e12, 10.0, sm)
    rho, q = IC.observables(res, r1, 1e12, 10.0)
    vc = np.full(NR, np.nan)
    for i, r in enumerate(R):
        if r >= 0.98*r1: continue
        h = 1e-3*r; d = 0.0
        for t, w in zip(TH_G, W_G):
            d += w*(-np.log(res.rho(r+h, t)/res.rho0)
                    + np.log(res.rho(r-h, t)/res.rho0))/(2*h)
        vc[i] = np.sqrt(r*res.sigma0**2*d)
    return np.log(rho), q, np.log(vc), r1


SMS = [0.1, 0.25, 0.5, 1.0]
D = {s: sidm(s) for s in SMS}
# best-fit CDM at sigma/m = 0.5, contraction allowed, M200 prior 0.1 dex
BEST = jnp.array([4.912e12, 1.92, 6.05e10, 0.806])
M = [np.asarray(v, float) for v in CDM(BEST)]

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.4))
C = dict(rho="#1f77b4", q="#d62728", vc="#2ca02c")

# A. residual map -----------------------------------------------------------
a = ax[0, 0]
res = []
for k, (lab, sig, col) in enumerate([(r"$\ln\rho$", SR, C["rho"]),
                                     (r"$q$ (axis ratio)", SQ, C["q"]),
                                     (r"$\ln v_c$", SV, C["vc"])]):
    rr = (D[0.5][k] - M[k]) / sig
    res.append(rr)
    a.plot(R, rr, "o-", color=col, label=lab, ms=4, lw=1.8)
a.axhline(0, color="k", lw=0.8); a.axvline(D[0.5][3], color="gray", ls=":", lw=1.5)
a.text(D[0.5][3]*1.05, -6.2, r"$r_1$", color="gray", fontsize=11)
a.set_xscale("log"); a.set_xlabel("r [kpc]")
a.set_ylabel(r"(SIDM $-$ best-fit CDM) / $\sigma$")
a.set_title("A. What CDM cannot fit\n"
            r"$\sigma/m=0.5$, CDM refits $M_{200},c,M_d,q_0$ with contraction",
            fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

# B. robustness of each channel to CDM's freedom ----------------------------
a = ax[0, 1]
VAR = ["0.1 ", "0.1 AC", "0.25", "0.25 AC", "0.5 ", "0.5 AC", "1.0 ", "1.0 AC",
       "0.1 P", "0.1 ACP", ".25 P", ".25 ACP", "0.5 P", "0.5 ACP", "1.0 P", "1.0 ACP"]
RHO_C = [749.5, 611.5, 644.9, 399.5, 612.4, 188.4, 750.1, 136.7,
         765.3, 829.1, 653.6, 534.3, 614.5, 242.1, 750.5, 142.0]
Q_C = [67.6, 67.6, 134.1, 134.2, 203.5, 203.5, 240.8, 240.8,
       67.6, 67.6, 134.1, 134.2, 203.5, 203.5, 240.8, 240.8]
x = np.arange(len(VAR))
a.plot(x, RHO_C, "o-", color=C["rho"], label=r"density  (spans $5.6\times$)")
a.plot(x, Q_C, "s-", color=C["q"], label="shape  (fixed by $\\sigma/m$ alone)")
for s, xs in zip(SMS, [(0, 1, 8, 9), (2, 3, 10, 11), (4, 5, 12, 13), (6, 7, 14, 15)]):
    a.annotate("", xy=(xs[0]-.4, 880), xytext=(xs[1]+.4, 880),
               arrowprops=dict(arrowstyle="-", color="gray", lw=.8))
a.set_xticks(x); a.set_xticklabels(VAR, rotation=70, fontsize=7)
a.set_ylabel(r"discriminating $\chi^2$")
a.set_title("B. Shape is immune to CDM's nuisance freedom\n"
            "16 CDM refits: +/- contraction, +/- mass prior", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

# C. the inversion ----------------------------------------------------------
a = ax[0, 2]
lab = ["density", "shape", r"$v_c$"]
est = [70.1, 1.3, 28.7]; dis = [53.7, 45.2, 1.1]
w = 0.36; xi = np.arange(3)
a.bar(xi-w/2, est, w, label="estimating $\\sigma/m$", color="#8c8c8c")
a.bar(xi+w/2, dis, w, label="discriminating vs CDM", color="#ff7f0e")
for i, (e, d) in enumerate(zip(est, dis)):
    a.text(i-w/2, e+1.4, f"{e:.1f}%", ha="center", fontsize=9)
    a.text(i+w/2, d+1.4, f"{d:.1f}%", ha="center", fontsize=9)
a.set_xticks(xi); a.set_xticklabels(lab); a.set_ylabel("share of total [%]")
a.set_title("C. The two questions invert\nshape: 1.3% for measuring, 45% for detecting",
            fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25, axis="y")

# D. saturation -------------------------------------------------------------
a = ax[1, 0]
for s in SMS:
    q = D[s][1]; k = np.isfinite(q)
    a.plot(R[k], q[k], "-", lw=2, label=f"$\\sigma/m={s}$")
a.axhline(0.7, color="k", ls="--", lw=1.6, label="CDM ($q=q_0$, flat)")
a.set_xscale("log"); a.set_xlabel("r [kpc]"); a.set_ylabel("axis ratio $q$")
a.set_title("D. Why: SIDM has a shape GRADIENT, CDM cannot\n"
            "the curves barely separate -> poor estimator; all differ from flat",
            fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

# E. clean information map --------------------------------------------------
a = ax[1, 1]
z = np.load(os.path.join(HERE, "J.npz")) if os.path.exists(
    os.path.join(HERE, "J.npz")) else np.load(
    "/tmp/claude-1003/-geir-data-scr-gabrielspace-jeans/"
    "c2b40140-c765-436f-8e08-af5382a50138/scratchpad/J.npz")
J, ok, SIGv = z["J"], z["ok"], z["SIG"]
Jl = J[ok]/SIGv[ok, None]; s = Jl[:, 2]
drop = np.zeros(Jl.shape[0]); F = Jl.T@Jl; base = np.linalg.inv(F)[2, 2]
for i in range(Jl.shape[0]):
    m = np.ones(Jl.shape[0], bool); m[i] = False
    drop[i] = np.linalg.inv(Jl[m].T@Jl[m])[2, 2] - base
raw = np.zeros(3*NR); raw[ok] = s**2
cln = np.zeros(3*NR); cln[ok] = drop
a.plot(R, raw[:NR]/raw[:NR].max(), "o--", color="#8c8c8c",
       label=r"raw $|\partial\ln\rho/\partial\ln(\sigma/m)|^2$")
a.plot(R, cln[:NR]/cln[:NR].max(), "o-", color=C["rho"],
       label="clean, after marginalising")
a.axvline(15.6, color="gray", ls=":", lw=1.5)
a.set_xscale("log"); a.set_xlabel("r [kpc]"); a.set_ylabel("normalised, density channel")
a.set_title("E. Raw and recoverable signal peak in DIFFERENT places\n"
            r"$r>r_1$ has zero direct signal yet 14% of the clean information",
            fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

# F. discrimination vs sigma/m ---------------------------------------------
a = ax[1, 2]
noac = [28.7, 28.1, 28.9, 32.2]; wac = [26.2, 23.2, 19.9, 19.7]
a.plot(SMS, noac, "o-", lw=2, label="CDM, no contraction")
a.plot(SMS, wac, "s-", lw=2, label="CDM + contraction (its best shot)")
a.plot(SMS, [np.sqrt(v) for v in [67.6, 134.1, 203.5, 240.8]], "^-", lw=2,
       color=C["q"], label="shape channel alone")
a.set_xscale("log"); a.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$]")
a.set_ylabel(r"discrimination $\sqrt{\chi^2}$  [$\sigma$]")
a.set_title("F. A bigger cross-section is NOT easier to detect\n"
            "a large core is well mimicked by low-concentration CDM", fontsize=10.5)
a.legend(frameon=False, fontsize=9); a.grid(alpha=.25)

fig.suptitle("SIDM vs CDM: where the differences are, and which survive "
             "refitting CDM   ($10^{12}\\,M_\\odot$ host, E5 baryons, "
             "$\\sigma_q=0.02$)", fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
out = os.path.join(HERE, "discrimination.png")
fig.savefig(out, dpi=135)
print("wrote", out)
print(f"channel chi2 at 0.5: " + " ".join(
    f"{n}={np.nansum(r**2):7.1f}" for n, r in zip(("rho", "q", "vc"), res)))
