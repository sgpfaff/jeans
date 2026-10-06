"""Can jeanie recover the known cross-section from an EAGLE SIDM halo?

The simulation knows the answer: SIDM1b was run at sigma/m = 1 cm^2/g, CDMb at
zero. Feed each halo's OWN baryons in as Phi_b, fit the matching radius to its
dark-matter profile, and read off sigma/m. If the model cannot recover a known
cross-section from a simulation, it will not recover one from a stream track.

    sigma/m = conversion / (rho1 * vrel * t_age),  vrel = 4 sigma0 / sqrt(pi)
"""
import os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = "/geir_data/scr/gabrielspace/jeans-fast/examples"
sys.path.insert(0, HERE); sys.path.insert(0, "/geir_data/scr/gabrielspace/jeans-fast/src")
os.chdir(HERE)
import simulations as sim
from jeanie.solver import solve_spherical
from jeanie.jaxsolver import nfw_boundary

GN, CONV, T_AGE = 4.302e-6, 4.685e9, 10.0


def nfw_rho(r, rhos, rs):
    x = r / rs
    return rhos / (x * (1 + x) ** 2)


def fit_concentration(r, rho, M200, r200, lo=0.5, hi=1.0):
    """c from the OUTER dark-matter profile, where SIDM and CDM agree.

    The range matters and is not a free choice. Fitting from 0.15 R200 includes
    radii the SIDM core has still modified, which biases c low, which biases
    r1, which biases the recovered cross-section -- measured on halo 2, that
    alone moved sigma/m from 0.995 to 0.125. [0.5, 1.0] R200 is where the NFW
    fit is actually good (per-point log misfit 0.0012 against 0.0156 at 0.3).
    """
    sel = np.isfinite(rho) & (rho > 0) & (r > lo * r200) & (r < hi * r200)
    rr, yy = r[sel], np.log(rho[sel])
    best = (np.inf, None)
    for c in np.linspace(2.0, 15.0, 131):
        rs = r200 / c
        rhos = M200 / (4 * np.pi * rs ** 3 * (np.log1p(c) - c / (1 + c)))
        resid = np.sum((np.log(nfw_rho(rr, rhos, rs)) - yy) ** 2)
        if resid < best[0]:
            best = (resid, c)
    return best[1]


def run(halo_id, model):
    f = sim.fit(halo_id, model=model)
    M200 = float(f.sph_data["M200"]); r200 = float(f.R200)
    d = f.sph_avg_dm_density()
    r, rho = d["r"], d["rho"]
    r, rho = np.asarray(r, float), np.asarray(rho, float)
    c = fit_concentration(r, rho, M200, r200)
    Phi_b = sim.compute_Phi_b_spherical(f.sph_data)

    # trial matching radii, as a fraction of r200
    out = []
    for frac in np.geomspace(0.004, 0.45, 40):
        r1 = frac * r200
        try:
            rho1, M1 = [float(v) for v in nfw_boundary(M200, c, r1)]
            res = solve_spherical(r1, rho1, M1, Phi_b=Phi_b, trajectory=True)
            if not res.success:
                continue
            sel = np.isfinite(rho) & (rho > 0) & (r > 0.004 * r200) & (r < r1)
            if sel.sum() < 6:
                continue
            pred = np.array([float(res.rho(x)) for x in r[sel]])
            chi = float(np.mean((np.log(pred) - np.log(rho[sel])) ** 2))
            sm = CONV / (rho1 * (4.0 * res.sigma0 / np.sqrt(np.pi)) * T_AGE)
            out.append((chi, r1, sm, res.r0, res.sigma0, int(sel.sum())))
        except Exception:
            continue
    if not out:
        return None
    out.sort()
    return dict(halo=halo_id, model=model, M200=M200, r200=r200, c=c,
                chi=out[0][0], r1=out[0][1], sigma_m=out[0][2],
                r0=out[0][3], sigma0=out[0][4], n=out[0][5])


print(f"{'model':8s}{'halo':>5}{'M200':>11}{'c':>6}{'r1/r200':>9}"
      f"{'r0 kpc':>9}{'sigma0':>9}{'sigma/m':>10}{'misfit':>9}")
for model in ("SIDM1b", "CDMb"):
    for h in (1, 2, 3, 4, 5):
        d = run(h, model)
        if d is None:
            print(f"{model:8s}{h:5d}   no solution found"); continue
        print(f"{d['model']:8s}{d['halo']:5d}{d['M200']:11.3e}{d['c']:6.2f}"
              f"{d['r1']/d['r200']:9.4f}{d['r0']:9.2f}{d['sigma0']:9.1f}"
              f"{d['sigma_m']:10.3f}{d['chi']:9.4f}")
print("\ntruth: SIDM1b was run at sigma/m = 1 cm^2/g; CDMb at 0.")
