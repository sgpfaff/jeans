"""Axisymmetric multipole expansion of a solved halo.

WHY THIS EXISTS. Not because galpy lacks one -- galpy 1.12 added
MultipoleExpansionPotential, it is C-backed, and jeanie.export uses it by
default. This module is the second implementation, and it earns its place
three ways: it needs no library at all, so shape diagnostics work in a bare
environment and on galpy 1.11, where the only basis expansion is SCF and
that does not converge on this profile at any order (40% median density
error at (N,L) = (8,6), 28% at (20,10)); it is a quadrature scheme rather
than a spline one, so it fails differently; and disagreeing with galpy is
what would catch either of them being wrong. On the comparison grid in
to_galpy this module is 2.5e-4 and galpy's is 2.1e-4.

Being the quadrature scheme is not a detail. Pinning a node either side of
the break at r1 is worth four orders of magnitude here -- and the same move
takes galpy's spline-based expansion to 2e+2, because two nodes a part in
1e9 apart are not something an interpolating spline can be asked to fit.

    rho(r, mu) = sum_l rho_l(r) P_l(mu),   mu = cos(theta)

    Phi_l(r) = -(4 pi G / (2l+1)) [ r^-(l+1) int_0^r rho_l r'^(l+2) dr'
                                  + r^l      int_r^inf rho_l r'^(1-l) dr' ]

The two radial integrals are accumulated by composite Simpson in ln r, with a
node pinned at every break radius so that no quadrature interval straddles
one. dPhi_l/dr is then exact rather than differenced -- the boundary terms of
the two integrals cancel identically -- which is what makes the Hermite
interpolant C1 and the forces smooth.

The density this is built from is DISCONTINUOUS in angle at r1: inside, the
baryonic disc flattens the isothermal solution through exp(-dPhi_b/sigma0^2);
outside, the CDM profile is evaluated on the squashed radius with q0, which
is 1 by default. That jump is a property of the model, not of this expansion.
A multipole represents it correctly (the radial integrals stay continuous
across a density jump); a basis expansion rings on it.

Convention: equatorial symmetry is assumed -- the density is even in z -- so
only even l contribute and odd harmonics are not computed.
"""
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicHermiteSpline
from scipy.special import eval_legendre, lpmv, roots_legendre

__all__ = ["MultipoleExpansion"]

GN = 4.302e-6          # kpc (km/s)^2 / Msun, jeanie's convention


def _odd(n):
    """Simpson wants an odd node count (an even number of intervals)."""
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def _tail_slope(r, f):
    """Local power-law index of f(r) from the last two nodes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log(np.abs(f[..., -1] / f[..., -2])) / np.log(r[-1] / r[-2])
    return np.where(np.isfinite(s), s, -3.0)


class MultipoleExpansion:
    """Expansion of rho(R, z) into spherical harmonics, with Phi and forces.

    Parameters
    ----------
    rho : callable
        rho(R, z) in Msun/kpc^3, broadcasting over arrays. jeanie's
        `density_fn(params)` is exactly this.
    rmin, rmax : float
        Radial range of the grid, in kpc. Inside rmin the potential is held
        at its rmin value (zero force), which is right for a cored profile
        and wrong for a cusp; outside rmax only the monopole is kept, as
        -G M_tot / r. Choose rmax well beyond the last radius you will use.
    lmax : int
        Highest harmonic. Only even l are computed.
    breaks : sequence of float
        Radii where the density is not smooth -- for a jeanie halo that is
        r1. Each gets its own grid node and no quadrature interval crosses
        it. This is what the SCF expansion cannot do.
    n_r : int
        Nodes per segment (segments are delimited by `breaks`).
    n_theta : int
        Gauss-Legendre nodes in mu on [0, 1].
    """

    def __init__(self, rho, rmin, rmax, lmax=8, breaks=(), n_r=257,
                 n_theta=24):
        self.lmax = int(lmax)
        self.ell = np.arange(0, self.lmax + 1, 2)
        self.rmin, self.rmax = float(rmin), float(rmax)

        edges = [self.rmin] + sorted(float(b) for b in breaks
                                     if self.rmin < b < self.rmax) + [self.rmax]
        # the upper side of a break starts just above it so that the density
        # callable takes its outer branch there, not its inner one
        # Each break is straddled by a pair of nodes a part in 1e9 either
        # side of it, so the density callable takes its inner branch on the
        # left segment and its outer branch on the right one whichever way
        # its own comparison is written (`r <= r1` or `r < r1`). Landing a
        # node exactly on the break would silently give one segment the
        # wrong branch for its endpoint, which is a few per cent on the
        # integral and invisible in the result.
        segs = []
        nb = len(edges) - 1
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            lo = lo if i == 0 else lo * (1.0 + 1e-9)
            hi = hi if i == nb - 1 else hi * (1.0 - 1e-9)
            segs.append(np.geomspace(lo, hi, _odd(n_r)))
        self.r = np.concatenate(segs)

        mu, wmu = roots_legendre(int(n_theta))          # on [-1, 1]
        mu, wmu = 0.5 * (mu + 1.0), 0.5 * wmu           # mapped to [0, 1]
        sin = np.sqrt(np.clip(1.0 - mu ** 2, 0.0, None))

        # one batched call: rho on the (n_r, n_theta) product grid
        RR = self.r[:, None] * sin[None, :]
        ZZ = self.r[:, None] * mu[None, :]
        vals = np.asarray(rho(RR, ZZ), float)

        # rho_l(r) = (2l+1) int_0^1 rho(r, mu) P_l(mu) dmu   (even l only)
        P = np.stack([eval_legendre(l, mu) for l in self.ell])      # (nl, nmu)
        wv = vals * wmu[None, :]                                    # (nr, nmu)
        self.rho_l = (2 * self.ell + 1)[:, None] * (P @ wv.T)       # (nl, nr)

        self._build(segs)

    # ------------------------------------------------------------------
    def _build(self, segs):
        r, ell = self.r, self.ell
        lv = ell[:, None].astype(float)
        f1 = self.rho_l * r[None, :] ** (lv + 3.0)      # integrand dr -> r dx
        f2 = self.rho_l * r[None, :] ** (2.0 - lv)

        # power-law tails below rmin and above rmax
        s_in = _tail_slope(r[:2][::-1], self.rho_l[:, :2][:, ::-1])
        rho_in = self.rho_l[:, 0]
        d1 = s_in + ell + 3.0
        I1_floor = np.where(d1 > 0, rho_in * r[0] ** (ell + 3.0) / np.where(d1 > 0, d1, 1.0), 0.0)

        s_out = _tail_slope(r, self.rho_l)
        rho_out = self.rho_l[:, -1]
        d2 = s_out + 2.0 - ell
        I2_tail = np.where(d2 < 0, -rho_out * r[-1] ** (2.0 - ell) / np.where(d2 < 0, d2, -1.0), 0.0)
        d3 = s_out + ell + 3.0
        self._M_tail = (float(-rho_out[0] * r[-1] ** 3.0 / d3[0])
                        if d3[0] < 0 else 0.0)

        I1 = np.zeros_like(f1)
        I2 = np.zeros_like(f2)
        # outward pass for the inner integral: its integrand rho_l r^(l+3)
        # rises with r, so the small contributions are added first.
        off = I1_floor.copy()
        pos = 0
        for seg in segs:
            sl = slice(pos, pos + len(seg))
            I1[:, sl] = cumulative_simpson(f1[:, sl], x=np.log(seg),
                                           initial=0.0, axis=1) + off[:, None]
            off = I1[:, sl][:, -1]
            pos += len(seg)
        # inward pass for the outer integral. Accumulating it instead as
        # (total - running) subtracts two numbers that differ by twenty
        # orders of magnitude once l >= 6, because rho_l r^(2-l) is then
        # dominated by the inner edge of the grid; harmonics came out 30-50%
        # wrong that way and converged to the wrong answer under refinement.
        off = I2_tail.copy()
        pos = len(self.r)
        for seg in segs[::-1]:
            sl = slice(pos - len(seg), pos)
            c = cumulative_simpson(f2[:, sl][:, ::-1], x=-np.log(seg)[::-1],
                                   initial=0.0, axis=1)
            I2[:, sl] = c[:, ::-1] + off[:, None]
            off = I2[:, sl][:, 0]
            pos -= len(seg)

        pre = -4.0 * np.pi * GN / (2 * ell + 1.0)
        Phi = pre[:, None] * (I1 / r[None, :] ** (lv + 1.0)
                              + r[None, :] ** lv * I2)
        dPhi = pre[:, None] * (-(lv + 1.0) * I1 / r[None, :] ** (lv + 2.0)
                               + lv * r[None, :] ** (lv - 1.0) * I2)

        u = np.log(r)
        self._Phi = [CubicHermiteSpline(u, Phi[i], dPhi[i] * r)
                     for i in range(len(ell))]
        # the force is the derivative of the interpolant, not a second
        # interpolant of the derivative: at the nodes the two agree by
        # construction, and in between only this one is consistent with the
        # Phi an orbit integrator is actually moving in. Differencing dPhi
        # separately would also blow up across the duplicated break node,
        # whose two copies are 1e-9 apart in ln r.
        self._dPhidu = [s.derivative() for s in self._Phi]
        self._Phi_edge = Phi[:, -1]
        self.M_total = 4.0 * np.pi * (I1[0, -1] + self._M_tail)
        self._Phi_in = Phi[:, 0]

    # ------------------------------------------------------------------
    def _harmonics(self, r):
        """Phi_l(r) and dPhi_l/dr(r), with the two extrapolations applied."""
        r = np.atleast_1d(np.asarray(r, float))
        rc = np.clip(r, self.rmin, self.rmax)
        u = np.log(rc)
        Phi = np.stack([s(u) for s in self._Phi])
        dPhi = np.stack([s(u) for s in self._dPhidu]) / rc[None, :]
        # outside rmax: keep the monopole only, as the exterior solution
        out = r > self.rmax
        if np.any(out):
            lv = self.ell[:, None].astype(float)
            fac = (self.rmax / r[None, :]) ** (lv + 1.0)
            Phi = np.where(out[None, :], self._Phi_edge[:, None] * fac, Phi)
            dPhi = np.where(out[None, :],
                            -(lv + 1.0) / r[None, :]
                            * self._Phi_edge[:, None] * fac, dPhi)
        # inside rmin: flat core, zero force
        ins = r < self.rmin
        if np.any(ins):
            Phi = np.where(ins[None, :], self._Phi_in[:, None], Phi)
            dPhi = np.where(ins[None, :], 0.0, dPhi)
        return Phi, dPhi

    # ------------------------------------------------------------------
    def potential(self, R, z):
        """Phi(R, z) in (km/s)^2."""
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        sh = R.shape
        R, z = R.ravel(), z.ravel()
        r = np.hypot(R, z)
        mu = np.where(r > 0, z / np.where(r > 0, r, 1.0), 1.0)
        Phi, _ = self._harmonics(r)
        P = np.stack([eval_legendre(l, mu) for l in self.ell])
        return np.sum(Phi * P, axis=0).reshape(sh)

    def forces(self, R, z):
        """(F_R, F_z) in (km/s)^2/kpc, i.e. -grad Phi."""
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        sh = R.shape
        R, z = R.ravel(), z.ravel()
        r = np.hypot(R, z)
        safe = np.where(r > 0, r, 1.0)
        mu = np.where(r > 0, z / safe, 1.0)
        sin = np.where(r > 0, R / safe, 0.0)
        Phi, dPhi = self._harmonics(r)
        P = np.stack([eval_legendre(l, mu) for l in self.ell])
        # sin(theta) dP_l/dmu = -P_l^1(mu) with the Condon-Shortley phase,
        # which is what scipy's lpmv returns -- and unlike dP_l/dmu it is
        # finite on the axis.
        P1 = np.stack([lpmv(1, l, mu) for l in self.ell])
        F_r = -np.sum(dPhi * P, axis=0)
        F_t = -np.sum(Phi * P1, axis=0) / safe
        FR = F_r * sin + F_t * mu
        Fz = F_r * mu - F_t * sin
        return FR.reshape(sh), Fz.reshape(sh)

    def density(self, R, z):
        """The expansion's own rho(R, z): sum_l rho_l(r) P_l(mu).

        This reads the stored harmonics rather than re-solving Poisson, so it
        measures the ANGULAR truncation at lmax and nothing else. Comparing
        it to the input density is the honest test of lmax.
        """
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        sh = R.shape
        R, z = R.ravel(), z.ravel()
        r = np.hypot(R, z)
        safe = np.where(r > 0, r, 1.0)
        mu = np.where(r > 0, z / safe, 1.0)
        u = np.log(np.clip(r, self.rmin, self.rmax))
        ur = np.log(self.r)
        rl = np.stack([np.interp(u, ur, self.rho_l[i])
                       for i in range(len(self.ell))])
        P = np.stack([eval_legendre(l, mu) for l in self.ell])
        return np.sum(rl * P, axis=0).reshape(sh)

    def harmonics(self, r):
        """(Phi_l(r), dPhi_l/dr(r)) for the even l this expansion carries.

        Exposed because comparing harmonic by harmonic is the only test of a
        multipole that does not go singular: Phi(R, z) passes through zeros
        of P_l where a relative error is meaningless.
        """
        return self._harmonics(r)

    def enclosed_mass(self, r):
        """M(<r) from the monopole alone, which is what sets the mean force."""
        r0 = np.asarray(r, float)
        r = np.atleast_1d(r0)
        _, dPhi = self._harmonics(r)
        out = dPhi[0] * r ** 2 / GN
        return out.item() if r0.ndim == 0 else out.reshape(r0.shape)
