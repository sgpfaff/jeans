"""Sample sigma/m directly, solving for r1, instead of sampling r1.

The first attempt sampled r1 and derived sigma/m, which failed: the corner
plot showed a tight c-r1 ridge and sigma/m almost a deterministic function of
r1. That extra freedom is not in the physics. r1 is DEFINED by there having
been about one scatter per particle per halo age at that radius,

    sigma/m = conversion / (rho1 * vrel * t_age),  vrel = 4 sigma0/sqrt(pi),

so given (M200, c, sigma/m) the matching radius is determined. Imposing that
removes the spurious degree of freedom. sigma/m is monotone increasing in r1
(measured: 0.013 to 2.9 as r1/R200 goes 0.008 to 0.23), so the inverse is a
bracketed bisection.
"""
import os, sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE="/geir_data/scr/gabrielspace/jeans-fast/examples"
sys.path.insert(0,HERE); sys.path.insert(0,"/geir_data/scr/gabrielspace/jeans-fast/src")
os.chdir(HERE)
import simulations as sim
from jeanie.solver import solve_spherical
from jeanie.jaxsolver import nfw_boundary
from jeanie.jaxouter import nfw_params

CONV, T_AGE, F_B = 4.685e9, 10.0, 0.156352
MODEL = sys.argv[1] if len(sys.argv)>1 else "SIDM1b"
HALO  = int(sys.argv[2]) if len(sys.argv)>2 else 2

f = sim.fit(HALO, model=MODEL)
M200_sim, R200 = float(f.sph_data["M200"]), float(f.R200)
d = f.sph_avg_dm_density()
r_all, rho_all = np.asarray(d["r"],float), np.asarray(d["rho"],float)
good = np.isfinite(rho_all)&(rho_all>0)&(r_all>0.005*R200)&(r_all<R200)
R_DAT, LOGRHO = r_all[good], np.log(rho_all[good])

# Pre-tabulate the simulation's baryon potential once. It is a vectorised
# scipy spline over a solve_ivp dense output, and _Problem calls it 400 times
# per construction; interpolating a fixed table is ~50x cheaper and is exact
# to the table's own resolution.
_pb_slow = sim.compute_Phi_b_spherical(f.sph_data)
_PB_R = np.concatenate([[0.0], np.geomspace(1e-3, 3.0*R200, 2000)])
_PB_V = np.array([float(_pb_slow(x)) for x in _PB_R])
def Phi_b(r):
    return np.interp(r, _PB_R, _PB_V)

print(f"[with Cautun contraction] {MODEL} halo {HALO}: M200={M200_sim:.3e} R200={R200:.0f} kpc, "
      f"{good.sum()} bins {R_DAT[0]:.1f}-{R_DAT[-1]:.0f} kpc; Phi_b tabulated")


def nfw_rho(r, rhos, rs):
    x = r/rs; return rhos/(x*(1.0+x)**2)


def nfw_M(r, rhos, rs):
    x = r/rs; return 4.0*np.pi*rhos*rs**3*(np.log1p(x) - x/(1.0+x))


# Baryon enclosed mass and its derivative, tabulated once. Same stellar mass
# profile that Phi_b is built from, so the contraction and the potential are
# consistent with each other.
_Mb_slow = sim.compute_Mb(f.sph_data)
_MB_R = np.concatenate([[0.0], np.geomspace(1e-3, 3.0*R200, 2000)])
_MB_V = np.array([float(_Mb_slow(x)) for x in _MB_R])
_ds = f.sph_avg_star_density()
_SR, _SRHO = np.asarray(_ds["r"], float), np.nan_to_num(np.asarray(_ds["rho"], float))

def Mb_of(r):
    return np.interp(r, _MB_R, _MB_V)

def dMb_of(r):
    return 4.0*np.pi*r**2*np.interp(r, _SR, _SRHO)


def cautun(r, rhos, rs):
    """Contracted dark-matter mass and density, Cautun et al. (2019).

        M_AC = M_CDM (0.45 + 0.38 [ (1-f_b)/f_b M_b/M_CDM + 1.16 ]^0.53)

    Closed form, so dM_AC/dr is taken analytically rather than by differencing
    a spline -- the package differences, and that is what put NaNs into its
    Einasto tail. rho_AC = M_AC'(r)/(4 pi r^2).
    """
    r = np.maximum(np.asarray(r, float), 1e-6)
    Mc, dMc = nfw_M(r, rhos, rs), 4.0*np.pi*r**2*nfw_rho(r, rhos, rs)
    k = (1.0 - F_B)/F_B
    x = k*Mb_of(r)/Mc
    g = 0.45 + 0.38*(x + 1.16)**0.53
    dg = 0.38*0.53*(x + 1.16)**(-0.47)
    dx = k*(dMb_of(r)*Mc - Mb_of(r)*dMc)/Mc**2
    M_ac = Mc*g
    dM_ac = dMc*g + Mc*dg*dx
    return M_ac, dM_ac/(4.0*np.pi*r**2)


def boundary(M200, c, r1):
    """(rho1, M1) at the matching radius, from the CONTRACTED outer halo."""
    rhos, rs = [float(v) for v in nfw_params(M200, c)]
    M1, rho1 = cautun(r1, rhos, rs)
    return float(rho1), float(M1)


def sigma_m_of(M200, c, r1):
    rho1, M1 = boundary(M200, c, r1)
    res = solve_spherical(r1, rho1, M1, Phi_b=Phi_b, trajectory=True)
    if not res.success:
        return None, None, None
    return CONV/(rho1*(4.0*res.sigma0/np.sqrt(np.pi))*T_AGE), res, rho1


def r1_from_sigma_m(M200, c, target, r200, n=18):
    """Bisect the monotone sigma/m(r1). Returns (r1, result) or (None, None)."""
    lo, hi = 1e-3*r200, 0.55*r200
    slo, _, _ = sigma_m_of(M200, c, lo)
    shi, _, _ = sigma_m_of(M200, c, hi)
    if slo is None or shi is None: return None, None
    if not (slo < target < shi): return None, None
    for _ in range(n):
        mid = np.sqrt(lo*hi)
        s, _, _ = sigma_m_of(M200, c, mid)
        if s is None: return None, None
        if s < target: lo = mid
        else: hi = mid
    r1 = np.sqrt(lo*hi)
    _, res, _ = sigma_m_of(M200, c, r1)
    return r1, res


LM0 = np.log10(M200_sim)

def log_prob(p):
    logM200, c, logsm, logs = p
    if not (LM0-0.15 < logM200 < LM0+0.15): return -np.inf, np.nan
    if not (2.0 < c < 25.0): return -np.inf, np.nan
    if not (-2.0 < logsm < 1.3): return -np.inf, np.nan        # sigma/m
    if not (-3.0 < logs < 0.5): return -np.inf, np.nan
    M200 = 10.0**logM200
    rhos, rs = [float(v) for v in nfw_params(M200, c)]
    r200 = rs*c
    r1, res = r1_from_sigma_m(M200, c, 10.0**logsm, r200)
    if r1 is None: return -np.inf, np.nan
    out = np.empty_like(R_DAT)
    inner = R_DAT <= r1
    out[inner] = [float(res.rho(float(x))) for x in R_DAT[inner]]
    _, rho_out = cautun(R_DAT[~inner], rhos, rs)
    out[~inner] = rho_out
    if not np.all(np.isfinite(out)) or np.any(out<=0): return -np.inf, np.nan
    s2 = (10.0**logs)**2
    ll = -0.5*np.sum((LOGRHO-np.log(out))**2/s2 + np.log(2*np.pi*s2))
    return ll, r1/r200


if __name__ == "__main__":
    import emcee
    import multiprocessing as mp
    # SPAWN, not fork: JAX is multithreaded and forking after touching it
    # deadlocks. These scripts used the default fork and survived it, which is
    # luck -- the identical pattern in bench/information hung three jobs for
    # fifty minutes with no output before the timeout killed them.
    CTX = mp.get_context("spawn")
    t0=time.time(); print("  timing one likelihood...", flush=True)
    lp,_ = log_prob(np.array([LM0, 6.0, 0.0, -1.0]))
    print(f"  one call: {time.time()-t0:.2f} s   logL={lp:.1f}", flush=True)
    nw, nd, nstep = 32, 4, int(sys.argv[3]) if len(sys.argv)>3 else 500
    # Every walker must START feasible. Half of them did not last time: the
    # sigma/m -> r1 bisection cannot bracket for every (M200, c, sigma/m), so
    # those walkers sat at -inf for the whole run, never moved, and their
    # parameter values then contaminated the posterior summary -- dragging the
    # quoted median from 0.050 to 0.170 and making a failed recovery look
    # consistent with the truth.
    rng = np.random.default_rng(0)
    seeds, tries = [], 0
    while len(seeds) < nw and tries < 20000:
        tries += 1
        cand = np.array([LM0 + 0.03*rng.standard_normal(),
                         rng.uniform(3.5, 12.0),
                         rng.uniform(-1.5, 0.8),
                         rng.uniform(-1.3, -0.4)])
        lp, _ = log_prob(cand)
        if np.isfinite(lp):
            seeds.append(cand)
    if len(seeds) < nw:
        raise SystemExit(f"only {len(seeds)} feasible seeds in {tries} tries")
    p0 = np.array(seeds)
    print(f"  {nw} feasible starting points found in {tries} draws", flush=True)
    t0=time.time()
    with CTX.Pool(26) as pool:
        s = emcee.EnsembleSampler(nw, nd, log_prob, pool=pool)
        s.run_mcmc(p0, nstep, progress=False)
    print(f"  sampled {nstep} steps in {time.time()-t0:.0f} s; "
          f"acceptance {np.mean(s.acceptance_fraction):.2f}")
    S="/tmp/claude-1003/-geir-data-scr-gabrielspace-jeans/c2b40140-c765-436f-8e08-af5382a50138/scratchpad"
    np.save(f"{S}/chain3_{MODEL}_{HALO}.npy", s.get_chain())
    np.save(f"{S}/blobs3_{MODEL}_{HALO}.npy", s.get_blobs())
    b = nstep//2
    fl = s.get_chain(discard=b, flat=True); bl = s.get_blobs(discard=b, flat=True)
    live = np.isfinite(bl)
    stuck = sum(1 for w in range(nw)
                if len(np.unique(s.get_chain(discard=b)[:, w, 2])) == 1)
    print(f"  {stuck} of {nw} walkers never moved; "
          f"{(~live).sum()} of {len(bl)} samples have no finite likelihood")
    if stuck:
        print("  -> summarising LIVE samples only")
    fl, bl = fl[live], bl[live]
    q = lambda v: np.percentile(v,[16,50,84])
    print(f"  log M200 : {q(fl[:,0])}")
    print(f"  c        : {q(fl[:,1])}")
    print(f"  r1/R200  : {q(bl[np.isfinite(bl)])}")
    print(f"  scatter  : {q(10**fl[:,3])}")
    lo,mid,hi = q(10**fl[:,2])
    print(f"  sigma/m  : {mid:.3f} +{hi-mid:.3f} -{mid-lo:.3f}   (truth "
          f"{'1' if MODEL=='SIDM1b' else '0'})")
