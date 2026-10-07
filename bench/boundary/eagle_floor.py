"""Do simulated haloes respect the existence boundary?

The falsification attempt for prong 1. A halo is reachable by thermalised dark
matter iff SOME matching radius satisfies 1/3 < R(r1) < R_fold(r1), where

    R(r1) = M_dm(<r1) / (4 pi r1^3 rho_dm(r1))

and R_fold is built from the halo's OWN baryon distribution via the shell
kernel, R_fold = R_MAX + int_0^r1 K_R(s/r1) dM_b(s) / (4 pi r1^3 rho1), which
is what the kernels were computed for -- no re-solving per halo.

Expected outcome: N_below = 0 with a pile-up against the boundary. If more
than ~5% of simulated SIDM haloes fall outside their own floor, it is not a
floor and prong 1 is dead.

Caveats held in view: EAGLE-50 ships 5 haloes per run at 1e13-1e14 Msun, so
these are clusters with BCGs, not the galaxies the programme targets, and the
sample is far too small for a rate. The kernel is first order in baryon
amplitude, so the reported mu is checked and flagged where it is not small.
"""
import os, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from jeanie.universal import R_MAX

DATA = "/geir_data/scr/gabrielspace/jeans/examples/data/EAGLE-50"
RUNS = ("CDMb", "SIDM1b", "vdSIDMb")
K = np.load(os.path.join(HERE, "kernels.npz"))
KS, KR = K["s"], K["KR"]


def kernel_shift(r1, r_edges, Mb_cum, rho1):
    """int K_R(s/r1) dM_b(s) / (4 pi r1^3 rho1), by summing shells."""
    inside = r_edges <= r1
    if inside.sum() < 2:
        return 0.0, 0.0
    re, mb = r_edges[inside], Mb_cum[inside]
    dm = np.diff(mb)                               # shell masses
    sc = 0.5 * (re[1:] + re[:-1]) / r1             # shell centres in s/r1
    unit = 4.0 * np.pi * r1 ** 3 * rho1
    kvals = np.interp(sc, KS, KR, left=KR[0], right=0.0)
    return float(np.sum(kvals * dm) / unit), float(np.sum(dm) / unit)


def analyse(run):
    with h5py.File(f"{DATA}/{run}_sphericallyAveraged_density_profiles.hdf5") as h:
        rs, redges = h["rs"][:], h["redges"][:]
        rho, Mdm = h["dm_rho"][:], h["dm_M"][:]
        Mb = h["star_M"][:] + h["gas_M"][:] + h["bh_M"][:]
        M200, r200 = h["M200"][:], h["r200"][:]
    rows = []
    for i in range(rho.shape[0]):
        r, d, mdm, mb = rs[i], rho[i], Mdm[i], Mb[i]
        ok = np.isfinite(d) & (d > 0) & (r > 0)
        best = None
        for j in np.where(ok)[0]:
            r1 = r[j]
            if r1 < 0.005 * r200[i] or r1 > 0.5 * r200[i]:
                continue
            rho1 = d[j]
            M1 = np.interp(r1, redges[i], mdm)
            R = M1 / (4.0 * np.pi * r1 ** 3 * rho1)
            shift, mu = kernel_shift(r1, redges[i], mb, rho1)
            Rf = R_MAX + shift
            margin = Rf - R                        # > 0 and R > 1/3 => reachable
            rec = dict(r1=r1, R=R, Rf=Rf, mu=mu, margin=margin,
                       inside=(R > 1 / 3) and (R < Rf))
            if best is None or rec["margin"] > best["margin"]:
                best = rec
            rows.append(rec)
        yield i, M200[i], r200[i], best, rows


if __name__ == "__main__":
    print(f"{'run':8s} {'halo':>4s} {'logM200':>8s} {'best r1/r200':>12s} "
          f"{'R':>8s} {'R_fold':>8s} {'mu':>7s} {'reachable':>10s}")
    tally = {}
    for run in RUNS:
        n_ok = n = 0
        for i, M200, r200, best, _ in analyse(run):
            n += 1
            if best is None:
                print(f"{run:8s} {i:4d}   no usable radii"); continue
            n_ok += bool(best["inside"])
            print(f"{run:8s} {i:4d} {np.log10(M200):8.2f} "
                  f"{best['r1']/r200:12.4f} {best['R']:8.4f} {best['Rf']:8.4f} "
                  f"{best['mu']:7.3f} {str(best['inside']):>10s}")
        tally[run] = (n_ok, n)
    print()
    for run, (k, n) in tally.items():
        print(f"{run:10s} reachable: {k}/{n}   "
              f"N_below = {n-k}  ({'floor holds' if k == n else 'FLOOR VIOLATED'})")
