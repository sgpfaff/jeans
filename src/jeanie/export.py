"""Export a solved halo as a galpy, agama or galax potential.

The solved halo is a set of callables -- rho(R,z), M(<r), g_r(r), Phi(r) --
and each library wants a different one of them, in different units. This
module does those conversions in one place, because getting them wrong is
silent: a potential with the wrong unit scaling still integrates orbits, it
just integrates the wrong ones.

    galpy   MultipoleExpansionPotential.from_density (new in galpy 1.12),
            or interpSphericalPotential from the radial force
    agama   Multipole built directly from the density callable
    galax   a JAX potential wrapping Phi, so gradients survive

BARYONS. jaxprofile's mass_fn, radial_acceleration_fn and potential_fn are
DARK MATTER ONLY -- the baryons shape the halo through Phi_b but are not
added to it. For an orbit or a stream you want the total, so `baryons=True`
is the DEFAULT here and adds the Miyamoto-Nagai disc that the solve already
carries in params. Omitting them is a silent error whose only tell is that an
orbit outside r1 has exactly zero gradient with respect to Md, a and b.

SHAPE. The halo is flattened inside r1 by the disc, with a gradient: q runs
from 0.62 at 3 kpc to 0.93 at 14 kpc for the reference halo. to_galpy's
"multipole" and to_agama keep that; to_galpy("spherical") and to_galax carry
the monopole only. Which is acceptable depends on what is being measured,
and for halo shape it is the whole measurement, so the default is the
multipole.

GALPY VERSION. galpy >= 1.12 is required for the default path:
MultipoleExpansionPotential arrived in 1.12. On 1.11 the only basis
expansion is SCFPotential, which does not converge on this profile at any
order (see to_galpy), and the fallback there is kind="multipole-ref" --
jeanie's own expansion, pure Python and slower, but library-free.

UNITS. jeanie works in kpc, Msun, km/s with G = 4.302e-6. Each exporter
converts into its target's convention and the round trip is asserted in
tests/test_export.py rather than assumed. agama and galax both use the
CODATA G instead, 4.30091727e-6, a systematic 2.5e-4; each exporter's
docstring says what it does about that, and the choice is not the same in
both, for reasons given there.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

from .jaxprofile import (density_fn, mass_fn, potential_fn,
                         radial_acceleration_fn, GN)
from .multipole import MultipoleExpansion

__all__ = ["callables", "multipole", "to_galpy", "to_agama", "to_galax"]


def callables(params, **kw):
    """The library-agnostic export: plain JAX functions.

    Returns a dict with rho(R,z), M(r), g(r) and Phi(r), all differentiable
    in `params`. Use this if your framework is not one of the three below --
    wrapping four callables is less work than fighting a converter.
    """
    return dict(rho=density_fn(params, **kw), M=mass_fn(params, **kw),
                g=radial_acceleration_fn(params, **kw),
                Phi=potential_fn(params, **kw))


def _mn(params, raw=False):
    """(Md, a, b) of the Miyamoto-Nagai disc carried in params.

    `raw=True` leaves them as whatever they already are, which is what the
    galax path needs: float() on a JAX tracer raises, so forcing it there
    would quietly rule out jax.grad through the exported potential.
    """
    if raw:
        return params[3], params[4], params[5]
    return float(params[3]), float(params[4]), float(params[5])


# ------------------------------------------------------------- multipole --
def multipole(params, lmax=8, rmin=None, rmax=None, n_r=257, n_theta=32,
              **kw):
    """The halo's own multipole expansion, framework-agnostic.

    Returns a `jeanie.multipole.MultipoleExpansion` with a grid node pinned
    either side of r1, where the density jumps: inside, the baryonic disc
    flattens the isothermal solution; outside, the CDM profile is spherical
    unless q0 is set.

    This is NOT what to_galpy uses -- galpy 1.12's own
    MultipoleExpansionPotential is better on every axis that matters for an
    orbit. It is kept because it needs no library at all, which makes it
    useful for shape diagnostics and, more to the point, because an
    independent second implementation is what caught two real errors here:
    a catastrophic cancellation in the outer radial integral at l >= 6, and
    the fact that pinning the break helps a quadrature scheme (four orders
    of magnitude) and ruins a spline one (galpy's density error goes to
    2e+2 if you try it there).
    """
    r1 = float(params[2])
    return MultipoleExpansion(
        density_fn(params, **kw),
        1e-3 * r1 if rmin is None else rmin,
        60.0 * r1 if rmax is None else rmax,
        lmax=lmax, breaks=(r1,), n_r=n_r, n_theta=n_theta)


def _as_scalar(a):
    a = np.asarray(a)
    return a.item() if a.ndim == 0 else a


def _batched_for_galpy(rho, rgrid, costheta_order, ro, D0):
    """A scalar density callable for galpy, served from one batched call.

    galpy's _compute_rho_lm walks `for r in rgrid: for ict in range(order)`
    and calls the density ONE POINT AT A TIME. Against a JAX function that
    is ~5 ms a call, so the default grid (800 x 20) takes 88 s to build --
    which rules the whole thing out for a run that generates many haloes.

    The points it will ask for are predictable, so they are all evaluated in
    a single vectorised call here and served from a table. The keys are
    bit-exact rather than rounded: galpy computes `sqrt(1 - ct**2)` from
    `leggauss(order)` and then `r * sintheta`, and the same operations on
    the same doubles give the same doubles. A key that misses anyway -- a
    different quadrature order inside galpy, a call from somewhere else --
    falls through to a live evaluation, so the table is a speedup and never
    a correctness assumption.
    """
    ct, _ = leggauss(int(costheta_order))
    st = np.sqrt(1.0 - ct ** 2)
    R = (rgrid[:, None] * st[None, :]).ravel()
    z = (rgrid[:, None] * ct[None, :]).ravel()
    vals = np.asarray(rho(R * ro, z * ro), float) / D0
    table = dict(zip(zip(R.tolist(), z.tolist()), vals.tolist()))

    def dens(Rq, zq):
        hit = table.get((Rq, zq)) if np.ndim(Rq) == 0 else None
        if hit is not None:
            return hit
        return np.asarray(rho(np.asarray(Rq) * ro,
                              np.asarray(zq) * ro), float) / D0

    dens.table = table            # for the test that asserts it is used
    return dens


def _galpy_multipole_potential(exp, ro, vo):
    """Wrap a MultipoleExpansion as a galpy Potential.

    galpy hands its potentials R and z in internal units and wants the
    potential in vo^2, forces in vo^2/ro and densities in vo^2/(G ro^2).
    """
    from galpy.potential import Potential

    P0, F0 = vo ** 2, vo ** 2 / ro
    D0 = (vo ** 2 * ro / GN) / ro ** 3

    class JeanieMultipolePotential(Potential):
        """Multipole expansion of a solved jeanie halo (dark matter only)."""

        def __init__(self):
            Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
            self.isNonAxi = False
            self.hasC = False
            self.hasC_dxdv = False
            self._expansion = exp

        def _evaluate(self, R, z, phi=0.0, t=0.0):
            return _as_scalar(exp.potential(np.asarray(R) * ro,
                                            np.asarray(z) * ro) / P0)

        def _Rforce(self, R, z, phi=0.0, t=0.0):
            return _as_scalar(exp.forces(np.asarray(R) * ro,
                                         np.asarray(z) * ro)[0] / F0)

        def _zforce(self, R, z, phi=0.0, t=0.0):
            return _as_scalar(exp.forces(np.asarray(R) * ro,
                                         np.asarray(z) * ro)[1] / F0)

        def _dens(self, R, z, phi=0.0, t=0.0):
            return _as_scalar(exp.density(np.asarray(R) * ro,
                                          np.asarray(z) * ro) / D0)

    return JeanieMultipolePotential()


# ----------------------------------------------------------------- galpy --
def to_galpy(params, ro=8.0, vo=220.0, kind="multipole", baryons=True,
             rmin=None, rmax=None, n_grid=301, N=12, L=10, lmax=8,
             n_r=None, n_theta=None, **kw):
    """galpy Potential (a combined one when baryons are included).

    Accuracies below are all measured on one grid -- 40 log-spaced radii from
    0.5 to 300 kpc crossed with mu = cos(theta) in {0, 0.3, 0.6, 0.9}, 160
    points -- as |rho_expansion / rho_input - 1|, because quoting a median
    over a point set chosen afterwards says more about the point set than
    about the expansion.

    kind="multipole" (default) is galpy's own MultipoleExpansionPotential,
    new in galpy 1.12, built from the density callable. 2.1e-4 median and
    3.0e-2 worst at lmax=8 with n_r=800; 1.0e-5 / 1.7e-2 at lmax=12 with
    n_r=1600. It has a C implementation, so an orbit integrates in C even
    though the density that built it is Python: 2000 steps over 3 Gyr in
    0.05 s at dE/|E| = 3e-14. The worst case is always the midplane inside
    r1, where a 0.28 kpc disc pinches the isothermal solution into an
    angular feature of its own.

    Do NOT try to pin a grid node on either side of r1 here. That is the
    right move for a quadrature scheme and the wrong one for a spline: galpy
    interpolates the radial coefficients, and two nodes a part in 1e9 apart
    take the density error to 2e+2. Use a denser uniform-in-log grid
    instead, which is what n_r does.

    kind="multipole-ref" is jeanie's own expansion (jeanie.multipole), at
    2.5e-4 median / 2.9e-2 worst. It is pure Python, so galpy cannot use its
    C integrator and falls back to leapfrog: measured 31x slower on the same
    orbit, 1.16 s against 0.04 s. It needs no galpy feature beyond the Potential base class,
    so it is the route on galpy 1.11, and it is the independent check on the
    default.

    kind="spherical" builds an interpSphericalPotential from the exact radial
    force. It round-trips galpy's own force against jeanie's to 1.2e-7, is
    the cheapest option, and DISCARDS the flattening entirely: as a density
    it is the monopole, 2.6e-2 median and 3.4e-1 worst. Fine for a mass
    model, wrong for anything that measures halo shape.

    kind="scf" is what galpy 1.11 had instead of a multipole, and it does not
    converge on this profile: 40% median and 450% worst at (N,L) = (8,6),
    28% and 400% at (20,10), so going up in order buys almost nothing. The
    fault is the basis, not the plumbing -- the same call reproduces a
    Hernquist density to 1e-15. A thermalised core matched to an Einasto
    envelope has a kink at r1 that a smooth Hernquist radial basis cannot
    follow, and no number of terms fixes a basis that is wrong. Kept only so
    the comparison can be reproduced.

    ZERO POINT. The three multipole routes here agree on forces to 2e-4 and
    disagree on Phi by a constant, because they continue the density past
    rmax differently: galpy sets it to zero, jeanie and agama extrapolate a
    power law. For this halo that constant is 3.2e3 (km/s)^2 between galpy
    and jeanie, which is the Einasto profile's own mass beyond 900 kpc. It
    cancels out of every force and every orbit, and it does not cancel out
    of an energy, so fix rmax deliberately if you are going to quote one.

    galpy works in internal units set by (ro, vo): lengths in ro kpc,
    velocities in vo km/s, so forces in vo^2/ro and potentials in vo^2.
    """
    from galpy.potential import (interpSphericalPotential, SCFPotential,
                                 MiyamotoNagaiPotential,
                                 scf_compute_coeffs_axi)
    r1 = float(params[2])
    rmin = 1e-3 * r1 if rmin is None else rmin
    rmax = 60.0 * r1 if rmax is None else rmax
    F0, P0 = vo ** 2 / ro, vo ** 2                    # force and potential units
    M0 = vo ** 2 * ro / GN                            # internal mass unit
    D0 = M0 / ro ** 3                                 # internal density unit

    if kind == "multipole":
        try:
            from galpy.potential import MultipoleExpansionPotential
        except ImportError as e:    # pragma: no cover - galpy < 1.12
            raise ImportError(
                "galpy >= 1.12 is needed for kind='multipole'; this galpy "
                "has no MultipoleExpansionPotential. Use "
                "kind='multipole-ref' for jeanie's own expansion, which "
                "needs nothing beyond the Potential base class."
            ) from e
        rho = density_fn(params, **kw)
        rgrid = np.geomspace(rmin, rmax, 800 if n_r is None else n_r) / ro
        # galpy's own default, reproduced here because the batched density
        # has to know which points it will be asked for
        order = max(20, lmax + 2) if n_theta is None else int(n_theta)
        dm = MultipoleExpansionPotential.from_density(
            _batched_for_galpy(rho, rgrid, order, ro, D0),
            L=lmax + 1,                       # galpy's L is max degree PLUS 1
            rgrid=rgrid, symmetry="axisymmetric",
            costheta_order=order, ro=ro, vo=vo)
    elif kind == "multipole-ref":
        dm = _galpy_multipole_potential(
            multipole(params, lmax=lmax, rmin=rmin, rmax=rmax,
                      n_r=257 if n_r is None else n_r,
                      n_theta=32 if n_theta is None else n_theta, **kw),
            ro, vo)
    elif kind == "spherical":
        g = radial_acceleration_fn(params, **kw)
        Phi = potential_fn(params, **kw)
        grid = np.geomspace(rmin, rmax, n_grid) / ro  # internal length
        dm = interpSphericalPotential(
            rforce=lambda x: float(g(x * ro)) / F0,
            rgrid=grid, Phi0=float(Phi(rmin)) / P0, ro=ro, vo=vo)
    elif kind == "scf":
        import warnings
        warnings.warn(
            "galpy's SCF basis does not converge on jeanie's profile: "
            "40%% median density error at (N,L)=(8,6) and 28%% at (20,10), "
            "from the kink at r1. Use kind='multipole' (the default, galpy "
            ">= 1.12) or kind='spherical' if you do not need the "
            "flattening.", RuntimeWarning, stacklevel=2)
        rho = density_fn(params, **kw)
        Acos, Asin = scf_compute_coeffs_axi(
            lambda R, z: float(rho(R * ro, z * ro)) / D0, N, L, a=r1 / ro)
        dm = SCFPotential(amp=1.0, Acos=Acos, Asin=Asin, a=r1 / ro,
                          ro=ro, vo=vo)
    else:
        raise ValueError("kind must be 'multipole', 'multipole-ref', "
                         "'spherical' or 'scf'; got %r" % kind)

    if not baryons:
        return dm
    Md, a, b = _mn(params)
    disc = MiyamotoNagaiPotential(amp=Md * 4.302e-6 / (vo ** 2 * ro),
                                  a=a / ro, b=b / ro, ro=ro, vo=vo)
    # '+' rather than a list: galpy deprecated lists of potentials and will
    # drop them after 1.13.
    return dm + disc


# ----------------------------------------------------------------- agama --
def to_agama(params, baryons=True, lmax=8, rmin=None, rmax=None,
             gridsize=None, **kw):
    """agama Potential, built as a Multipole over the density callable.

    agama's unit system is global, so this sets it to (Msun, kpc, km/s) --
    jeanie's convention -- which will affect any other agama object in the
    session. That is agama's design, not a choice made here, and it is the
    usual source of silently wrong results when mixing sources.

    agama solves Poisson with the CODATA G, while jeanie's halo was solved
    with G = 4.302e-6, so everything this returns -- halo and disc alike --
    is uniformly 2.5e-4 away from jeanie. Measured: the forces agree with
    jeanie's own multipole to a median 2.1e-4, which is that ratio and
    almost nothing else. It is left as it is, because applying one G
    throughout is the self-consistent choice; see to_galax, where the two
    components would otherwise have disagreed with each other.

    `gridsize` is agama's radial grid, and the defaults are not its best.
    On the grid described in to_galpy, density error against the input
    callable is 1.3e-3 median / 2.5e-2 worst at lmax=8 on the auto grid, and
    2.3e-4 / 1.4e-2 at lmax=12 with gridsize=40 -- which is jeanie's own
    multipole to within the noise. So agama is not behind here once it is
    asked properly; it is behind at its defaults, by about 5x in the median.
    Raise both if the shape is the measurement.
    """
    import agama
    agama.setUnits(mass=1, length=1, velocity=1)      # Msun, kpc, km/s
    r1 = float(params[2])
    rho = density_fn(params, **kw)

    def dens(xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        R = np.hypot(xyz[:, 0], xyz[:, 1])
        return np.asarray(rho(R, xyz[:, 2]), float)

    kw_ag = dict(type="Multipole", density=dens, symmetry="axisymmetric",
                 lmax=lmax, rmin=1e-3 * r1 if rmin is None else rmin,
                 rmax=60.0 * r1 if rmax is None else rmax)
    if gridsize:
        kw_ag["gridSizeR"] = int(gridsize)
    dm = agama.Potential(**kw_ag)
    if not baryons:
        return dm
    Md, a, b = _mn(params)
    disc = agama.Potential(type="MiyamotoNagai", mass=Md,
                           scaleradius=a, scaleheight=b)
    return agama.Potential(dm, disc)


# ----------------------------------------------------------------- galax --
def to_galax(params, baryons=True, units="galactic", **kw):
    """galax potential wrapping jeanie's JAX Phi, so gradients survive.

    The point of a galax export rather than a galpy one is differentiability
    end to end: Phi here is the same traced function jaxprofile produces, so
    d(orbit)/d(halo parameters) works without finite differences.

    SPHERICALLY AVERAGED. potential_fn is a function of r alone, so this
    export carries the monopole and drops the flattening -- measured q = 0.62
    at 3 kpc rising to 0.93 at 14 kpc for the reference halo. The flattening
    is in the multipole, which is built with scipy and is therefore not
    differentiable in the halo parameters; you can have shape (to_galpy or
    to_agama) or gradients (here), not yet both.

    G. galax uses the CODATA value; jeanie uses G = 4.302e-6, and that is
    the G the halo was solved with. A galax MiyamotoNagaiPotential would
    therefore sit 2.5e-4 deep beside it -- small, but a mismatch BETWEEN the
    two components of one potential, which is worse than being 2.5e-4 away
    from CODATA as a whole. So the disc here is jeanie's own analytic
    Miyamoto-Nagai rather than galax's, carrying the same G as the halo. It
    stays differentiable, and nothing has to be relabelled: galax's class
    would have needed m_tot scaled by 1.000252 to agree, i.e. reporting a
    mass it was never given.

    agama takes the other branch -- it applies CODATA to jeanie's densities,
    so its export is internally consistent and uniformly 2.5e-4 away from
    jeanie. galpy has no offset at all; its amplitude conversion carries
    jeanie's G explicitly and the test asserts it to 1e-10.

    UNITS. galax's "galactic" system measures speed in kpc/Myr, not km/s, so
    its potentials are in (kpc/Myr)^2 while jeanie's are in (km/s)^2. The two
    differ by 9.56e5. Returning jeanie's number unconverted leaves the halo
    that factor too shallow next to a galax MiyamotoNagaiPotential built from
    the same params, and the orbit still integrates -- it just orbits the
    disc. The conversion below is taken from the unit system passed in rather
    than hard-coded, so a different `units` stays correct.
    """
    try:
        import jax.numpy as jnp
        import galax.potential as gp
        from unxt.unitsystems import unitsystem
        import astropy.units as apu
    except ImportError as e:        # pragma: no cover - optional dependency
        raise ImportError(
            "to_galax needs galax (pip install galax). For any other "
            "framework use jeanie.export.callables(params), which returns "
            "plain JAX rho/M/g/Phi."
        ) from e

    usys = unitsystem(units)
    to_kpc = float((1.0 * apu.Unit(str(usys["length"]))).to_value(apu.kpc))
    from_kms2 = float(((apu.km / apu.s) ** 2)
                      .to(apu.Unit(str(usys["speed"])) ** 2))

    Phi = potential_fn(params, **kw)
    Md, a, b = _mn(params, raw=True)

    class JeanieHalo(gp.AbstractSinglePotential):
        """Spherically averaged jeanie halo, differentiable in params."""

        def _potential(self, q, t):                   # noqa: D401
            r = jnp.sqrt(jnp.sum(jnp.square(q), axis=-1)) * to_kpc
            return Phi(r) * from_kms2

    class JeanieDisc(gp.AbstractSinglePotential):
        """The same Miyamoto-Nagai the halo was solved in, same G."""

        def _potential(self, q, t):                   # noqa: D401
            R2 = jnp.sum(jnp.square(q[..., :2]), axis=-1) * to_kpc ** 2
            z = q[..., 2] * to_kpc
            zb = a + jnp.sqrt(z ** 2 + b ** 2)
            return -GN * Md / jnp.sqrt(R2 + zb ** 2) * from_kms2

    try:
        dm = JeanieHalo(units=usys)
    except Exception as e:          # pragma: no cover - API drift
        raise RuntimeError(
            "galax's potential interface has changed; build one yourself "
            "from jeanie.export.callables(params)['Phi'], which is the same "
            "traced JAX function this wrapper uses. Original error: %s" % e
        ) from e
    if not baryons:
        return dm
    return gp.CompositePotential(halo=dm, disc=JeanieDisc(units=usys))
