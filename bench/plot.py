"""Figures from the benchmark JSON.

Everything plotted here is read from bench/results/*.json, so a figure can
never disagree with the recorded measurement. Uncertainty is drawn wherever it
was measured: the speed-up bars carry the interquartile range of both the
package and the reduced solver, because a bare ratio of two single samples on
a shared machine is not a result.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

C = {"pkg": "#B2182B", "fast": "#2166AC", "e2e": "#8EB8DC",
     "ref": "#6B7280", "good": "#1B7837", "warn": "#E08214", "mute": "#D6DBE3"}
W1, W2 = 3.5, 7.2


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.subplots_adjust(wspace=0.34)
    p = os.path.join(FIGS, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name} ({os.path.getsize(p)/1024:.0f} kB)")


# ----------------------------------------------------------------- speed-up
def fig_speedup(d):
    rows = d["cases"]
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))[::-1]

    so = np.array([r["speedup_solver_only"]["median"] for r in rows])
    so_lo = np.array([r["speedup_solver_only"]["lo"] for r in rows])
    so_hi = np.array([r["speedup_solver_only"]["hi"] for r in rows])
    e2 = np.array([r["speedup_end_to_end"]["median"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(W2, 0.46 * len(rows) + 2.3),
                           gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) absolute cost
    a = ax[0]
    for i, (yy, r) in enumerate(zip(y, rows)):
        p, f = r["package"], r["fast_solver"]
        a.barh(yy + 0.20, p["median_s"] * 1e3, height=0.38, color=C["pkg"],
               xerr=[[(p["median_s"] - p["q1_s"]) * 1e3],
                     [(p["q3_s"] - p["median_s"]) * 1e3]],
               error_kw=dict(ecolor="#5A1015", lw=0.9, capsize=2))
        a.barh(yy - 0.20, f["median_s"] * 1e3, height=0.38, color=C["fast"],
               xerr=[[(f["median_s"] - f["q1_s"]) * 1e3],
                     [(f["q3_s"] - f["median_s"]) * 1e3]],
               error_kw=dict(ecolor="#10335A", lw=0.9, capsize=2))
    a.set_xscale("log")
    a.set_yticks(y)
    a.set_yticklabels(names)
    a.set_xlabel("wall clock per halo [ms]")
    a.set_title("(a) cost, median with interquartile range", loc="left")
    a.set_xlim(5e-3, 2e5)
    a.legend(handles=[Patch(color=C["pkg"], label="package"),
                      Patch(color=C["fast"], label="reduced solver")],
             loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=2)
    a.grid(axis="x", color=C["mute"], lw=0.5, alpha=0.6, zorder=0)
    a.set_axisbelow(True)

    # (b) speed-up
    b = ax[1]
    b.barh(y + 0.20, so, height=0.38, color=C["fast"],
           xerr=[so - so_lo, so_hi - so], error_kw=dict(ecolor="#10335A", lw=0.9, capsize=2))
    b.barh(y - 0.20, e2, height=0.38, color=C["e2e"])
    for yy, v, r in zip(y, so, rows):
        # Spell the interval out: on a log axis the error bar is sub-pixel
        # wherever the spread is a few per cent, which is most cases.
        band = 100 * (r["speedup_solver_only"]["hi"] /
                      r["speedup_solver_only"]["lo"] - 1.0)
        b.text(v * 1.35, yy + 0.20, f"{v:,.0f}×  ±{band/2:.0f}%", va="center",
               fontsize=7.2, color="#10335A")
    b.set_xscale("log")
    b.set_yticks(y)
    b.set_yticklabels([])
    b.set_xlabel("speed-up")
    b.set_title("(b) speed-up", loc="left")
    b.set_xlim(1, max(so_hi) * 60)
    b.axvline(1.0, color=C["ref"], lw=0.8)
    b.legend(handles=[Patch(color=C["fast"], label="interior solve"),
                      Patch(color=C["e2e"], label="including outer halo")],
             loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=2)
    b.grid(axis="x", color=C["mute"], lw=0.5, alpha=0.6, zorder=0)
    b.set_axisbelow(True)
    save(fig, "bench1_speedup.png")


# ------------------------------------------------------- accuracy vs cost
def fig_pareto(cost, cases):
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.95))
    pkg_err = {}
    for r in cases["cases"]:
        if r["name"] == "2D L=[0,2], no baryons":
            pkg_err["no baryons"] = (r["package"]["median_s"] * 1e3,
                                     r["accuracy"]["r0_rel"],
                                     r["package"]["q1_s"] * 1e3,
                                     r["package"]["q3_s"] * 1e3)
        if r["name"] == "2D L=[0,2] + disc":
            pkg_err["MN disc"] = (r["package"]["median_s"] * 1e3,
                                  r["accuracy"]["r0_rel"],
                                  r["package"]["q1_s"] * 1e3,
                                  r["package"]["q3_s"] * 1e3)

    for k, (label, col) in enumerate([("no baryons", C["good"]), ("MN disc", C["fast"])]):
        rows = cost["series"].get(label)
        if not rows:
            continue
        a = ax[k]
        t = np.array([r["time"]["median_s"] for r in rows]) * 1e3
        q1 = np.array([r["time"]["q1_s"] for r in rows]) * 1e3
        q3 = np.array([r["time"]["q3_s"] for r in rows]) * 1e3
        e = np.array([max(r["r0_rel_to_converged"], 1e-16) for r in rows])
        ns = [r["n_steps"] for r in rows]
        # Horizontal bars only: wall clock is stochastic and carries a measured
        # interquartile range, while the accuracy axis is a deterministic
        # computation that returns identical bits on a rerun. Drawing a vertical
        # bar would invent an uncertainty that does not exist.
        a.set_xscale("log")
        a.set_yscale("log")
        a.errorbar(t, e, xerr=[t - q1, q3 - t], fmt="o-", color=col, ms=5,
                   lw=1.4, elinewidth=1.1, capsize=2.5, zorder=3)
        for ti, ei, n in zip(t, e, ns):
            a.annotate(f"{n}", (ti, ei), textcoords="offset points",
                       xytext=(5, 5), fontsize=6.8, color=col)
        if label in pkg_err:
            pt, pe, pq1, pq3 = pkg_err[label]
            a.errorbar([pt], [pe], xerr=[[pt - pq1], [pq3 - pt]], fmt="s",
                       color=C["pkg"], ms=7, elinewidth=1.1, capsize=2.5, zorder=4)
            a.annotate("package\n(default grid)", (pt, pe),
                       textcoords="offset points", xytext=(-10, -22),
                       fontsize=7.2, color=C["pkg"], ha="right")
            a.axhline(pe, color=C["pkg"], lw=0.9, ls="--", alpha=0.7)
        a.set_xlabel("wall clock per halo [ms]")
        a.set_title(f"({'ab'[k]}) 2D $L{{=}}[0,2]$, {label}", loc="left")
        a.grid(color=C["mute"], lw=0.5, alpha=0.6)
        a.set_axisbelow(True)
    ax[0].set_ylabel(r"$|\Delta r_0 / r_0|$ vs. converged")
    save(fig, "bench2_pareto.png")


# ---------------------------------------------------------- grid refinement
def fig_gridref(d):
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    cols = [C["good"], C["fast"], C["warn"]]
    for (label, s), col in zip(d["series"].items(), cols):
        N = np.array([r["r_grid"] for r in s["rows"]], float)
        e = np.array([r["r0_rel"] for r in s["rows"]])
        ax[0].loglog(N, e, "o-", color=col, ms=4.5, lw=1.3, label=label)
        orders = [r.get("order") for r in s["rows"][1:]]
        mid = np.sqrt(N[:-1] * N[1:])
        ax[1].semilogx(mid, orders, "o-", color=col, ms=4.5, lw=1.3, label=label)
    Ng = np.array([42.0, 950.0])
    ref = max(np.array([r["r0_rel"] for r in list(d["series"].values())[0]["rows"]]))
    ax[0].loglog(Ng, ref * 2.2 * (Ng / 50.0) ** -2, ":", color=C["ref"], lw=1.1)
    ax[0].text(260, ref * 0.12, r"$\propto N^{-2}$", color=C["ref"],
               fontsize=9, rotation=-30)
    ax[0].axvline(200, color=C["ref"], lw=0.7, ls="--")
    ax[0].text(208, ref * 1.1, "package\ndefault", fontsize=7, color=C["ref"])
    ax[0].set_xlabel("package radial grid $N$")
    ax[0].set_ylabel(r"$|r_0^{\rm pkg}/r_0^{\rm fast} - 1|$")
    ax[0].set_title("(a) the gap closes with no floor", loc="left")
    ax[0].text(0.03, 0.03, "deterministic: no timing involved", transform=ax[0].transAxes,
               fontsize=6.6, color=C["ref"], style="italic")
    ax[0].legend(loc="lower left")
    ax[0].grid(color=C["mute"], lw=0.5, alpha=0.6)
    ax[0].set_axisbelow(True)

    ax[1].axhline(2.0, color=C["ref"], lw=1.0, ls="--")
    ax[1].text(62, 1.735, "exact second order", fontsize=7.5, color=C["ref"])
    ax[1].set_ylim(1.7, 2.3)
    ax[1].set_xticks([70, 100, 150, 250, 400, 600])
    ax[1].set_xticklabels(["70", "100", "150", "250", "400", "600"])
    ax[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[1].set_xlabel(r"$N$ (geometric mean of the pair)")
    ax[1].set_ylabel("local convergence order")
    ax[1].set_title("(b) order is 2", loc="left")
    ax[1].grid(color=C["mute"], lw=0.5, alpha=0.6)
    ax[1].set_axisbelow(True)
    save(fig, "bench3_gridref.png")


# --------------------------------------------------------------- FD ladder
def fig_ladder(d):
    L = d["ladder"]
    s = np.array(L["steps"])
    series = [("package_spherical", "package, pre-D6 convention", C["pkg"], "o"),
              ("package_exact", "package, D6 fixed", C["warn"], "^"),
              ("fast", "reduced solver", C["fast"], "s")]
    plateau = float(np.median(L["fast"][3:]))

    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.95))

    a = ax[0]
    for key, lab, col, mk in series:
        a.semilogx(s, np.array(L[key], float), mk + "-", color=col, ms=4.3,
                   lw=1.3, label=lab)
    a.axhline(plateau, color=C["ref"], lw=0.9, ls="--", zorder=0)
    a.invert_xaxis()
    a.set_xlabel("relative finite-difference step $s$")
    a.set_ylabel(r"$d\log r_0\,/\,d\log M_{200}$")
    a.set_title("(a) the derivative", loc="left")
    a.text(0.03, 0.03, "deterministic: no timing involved", transform=a.transAxes,
           fontsize=6.6, color=C["ref"], style="italic")
    a.grid(color=C["mute"], lw=0.5, alpha=0.6)
    a.set_axisbelow(True)
    a.annotate("sign flips", xy=(3e-7, -0.341), xytext=(4e-4, -0.30),
               fontsize=7.5, color=C["pkg"], ha="left",
               arrowprops=dict(arrowstyle="->", color=C["pkg"], lw=0.8))
    a.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    b = ax[1]
    # Each series is compared against its OWN value at s=1e-3, so this shows
    # step-size instability alone and not the (physical) offset between the
    # two baryon conventions.
    i_ref = L["steps"].index(1e-3)
    for key, lab, col, mk in series:
        v = np.array(L[key], float)
        dev = np.abs(v / v[i_ref] - 1.0)
        b.loglog(s, np.maximum(dev, 1e-17), mk + "-", color=col, ms=4.3,
                 lw=1.3, label=lab)
    b.invert_xaxis()
    b.axhline(1e-2, color=C["ref"], lw=0.8, ls=":")
    b.text(1.4e-7, 1.4e-2, "1% error", fontsize=7, color=C["ref"], ha="left")
    b.set_ylim(1e-17, 2e1)
    b.set_xlabel("relative finite-difference step $s$")
    b.set_ylabel(r"relative departure from own $s{=}10^{-3}$ value")
    b.set_title("(b) step-size instability", loc="left")
    b.legend(loc="upper left")
    b.grid(color=C["mute"], lw=0.5, alpha=0.6)
    b.set_axisbelow(True)
    save(fig, "bench4_ladder.png")


# ------------------------------------------------------- measurement scatter
def fig_noise(d):
    """Every timing sample, normalised to its own case median.

    The error bars in fig_speedup are real but sub-pixel: a 2% interquartile
    range on a log axis spanning seven decades is 0.03% of the axis width, so
    they read as "no uncertainty was measured". This figure exists so the
    scatter is actually visible. Each dot is one run.
    """
    rows = d["cases"]
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(1, 2, figsize=(W2, 0.46 * len(rows) + 2.0),
                           gridspec_kw={"width_ratios": [1.5, 1]})

    a = ax[0]
    for yy, r in zip(y, rows):
        for key, off, col in (("package", 0.20, C["pkg"]),
                              ("fast_solver", -0.20, C["fast"])):
            t = r[key]
            sm = np.array(t.get("samples_s") or [t["median_s"]], float)
            ratio = sm / t["median_s"]
            jit = rng.uniform(-0.055, 0.055, size=len(ratio))
            a.plot(ratio, np.full_like(ratio, yy + off) + jit, "o", ms=3.4,
                   color=col, alpha=0.75, mew=0)
            a.plot([t["q1_s"] / t["median_s"], t["q3_s"] / t["median_s"]],
                   [yy + off, yy + off], "-", color=col, lw=1.6, alpha=0.45,
                   solid_capstyle="round", zorder=0)
    a.axvline(1.0, color=C["ref"], lw=0.9)
    a.set_yticks(y)
    a.set_yticklabels(names)
    a.set_xlabel("run time / median for that case")
    a.set_title("(a) every sample, normalised per case", loc="left")
    a.grid(axis="x", color=C["mute"], lw=0.5, alpha=0.6)
    a.set_axisbelow(True)
    a.legend(handles=[Patch(color=C["pkg"], label="package (5 runs)"),
                      Patch(color=C["fast"], label="reduced solver (21 runs)")],
             loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2)

    b = ax[1]
    pk = np.array([100 * (r["package"]["q3_s"] / r["package"]["q1_s"] - 1) for r in rows])
    fa = np.array([100 * (r["fast_solver"]["q3_s"] / r["fast_solver"]["q1_s"] - 1) for r in rows])
    b.barh(y + 0.19, pk, height=0.36, color=C["pkg"])
    b.barh(y - 0.19, fa, height=0.36, color=C["fast"])
    for yy, v in zip(y, np.maximum(pk, fa)):
        b.text(v + 0.6, yy, f"{v:.0f}%", va="center", fontsize=7, color=C["ref"])
    b.set_yticks(y)
    b.set_yticklabels([])
    b.set_xlabel("interquartile spread [%]")
    b.set_title("(b) spread", loc="left")
    b.set_xlim(0, max(pk.max(), fa.max()) * 1.25)
    b.grid(axis="x", color=C["mute"], lw=0.5, alpha=0.6)
    b.set_axisbelow(True)
    save(fig, "bench5_noise.png")


if __name__ == "__main__":
    print("plotting ->", FIGS)
    cases, cost = load("cases"), load("cost")
    gridref, ladder = load("gridref"), load("ladder")
    if cases:
        fig_speedup(cases)
        fig_noise(cases)
    if cost and cases:
        fig_pareto(cost, cases)
    if gridref:
        fig_gridref(gridref)
    if ladder:
        fig_ladder(ladder)
