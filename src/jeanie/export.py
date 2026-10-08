"""Export a solved halo as a galpy, agama or galax potential.

The solved halo is a set of callables -- rho(R,z), M(<r), g_r(r), Phi(r) --
and each library wants a different one of them, in different units. This
module does those conversions in one place, because getting them wrong is
silent: a potential with the wrong unit scaling still integrates orbits, it
just integrates the wrong ones.

    galpy   interpSphericalPotential from the radial force, or SCFPotential
            from the density for the axisymmetric case
    agama   Multipole built directly from the density callable
    galax   a JAX potential wrapping Phi, so gradients survive

BARYONS. jaxprofile's mass_fn, radial_acceleration_fn and potential_fn are
DARK MATTER ONLY -- the baryons shape the halo through Phi_b but are not
added to it. For an orbit or a stream you want the total, so `baryons=True`
is the DEFAULT here and adds the Miyamoto-Nagai disc that the solve already
carries in params. Omitting them is a silent error whose only tell is that an
orbit outside r1 has exactly zero gradient with respect to Md, a and b.

UNITS. jeanie works in kpc, Msun, km/s with G = 4.302e-6. Each exporter
converts into its target's convention and the round trip is asserted in
tests/test_export.py rather than assumed.
"""
import numpy as np

from .jaxprofile import (density_fn, mass_fn, potential_fn,
                         radial_acceleration_fn, GN)

__all__ = ["callables", "to_galpy", "to_agama", "to_galax"]


def callables(params, **kw):
    """The library-agnostic export: plain JAX functions.

    Returns a dict with rho(R,z), M(r), g(r) and Phi(r), all differentiable
    in `params`. Use this if your framework is not one of the three below --
    wrapping four callables is less work than fighting a converter.
    """
    return dict(rho=density_fn(params, **kw), M=mass_fn(params, **kw),
                g=radial_acceleration_fn(params, **kw),
                Phi=potential_fn(params, **kw))


def _mn(params):
    """(Md, a, b) of the Miyamoto-Nagai disc carried in params."""
    return float(params[3]), float(params[4]), float(params[5])


# ----------------------------------------------------------------- galpy --
def to_galpy(params, ro=8.0, vo=220.0, kind="spherical", baryons=True,
             rmin=None, rmax=None, n_grid=301, N=12, L=10, **kw):
    """galpy Potential (a combined one when baryons are included).

    kind="spherical" builds an interpSphericalPotential from the exact radial
    force. It round-trips galpy's own force against jeanie's to 1.2e-7 and is
    the recommended path.

    kind="scf" keeps the flattening but DOES NOT CONVERGE on this profile and
    warns accordingly. Measured density error against jeanie: 49% median at
    (N,L) = (8,6), 27% at (20,10), 22% at (30,12), bottoming out near 8.5%
    median / 22% max with the basis scale stretched to 45 kpc. The fault is
    the basis, not the plumbing -- the same call reproduces a Hernquist
    density to 1e-15 -- because a thermalised core matched to an Einasto
    envelope has a kink at r1 that a smooth Hernquist expansion cannot
    follow. For a flattened halo use to_agama, whose Multipole is built from
    the density directly and has no such basis.

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

    if kind == "spherical":
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
            "measured density error 22%% median at (N,L)=(30,12), from the "
            "kink at r1. Use kind='spherical' (1.2e-7) or to_agama for a "
            "flattened halo.", RuntimeWarning, stacklevel=2)
        rho = density_fn(params, **kw)
        M0 = vo ** 2 * ro / GN                        # internal mass unit
        D0 = M0 / ro ** 3                             # internal density unit
        Acos, Asin = scf_compute_coeffs_axi(
            lambda R, z: float(rho(R * ro, z * ro)) / D0, N, L, a=r1 / ro)
        dm = SCFPotential(amp=1.0, Acos=Acos, Asin=Asin, a=r1 / ro,
                          ro=ro, vo=vo)
    else:
        raise ValueError("kind must be 'spherical' or 'scf'; got %r" % kind)

    if not baryons:
        return dm
    Md, a, b = _mn(params)
    disc = MiyamotoNagaiPotential(amp=Md * 4.302e-6 / (vo ** 2 * ro),
                                  a=a / ro, b=b / ro, ro=ro, vo=vo)
    # '+' rather than a list: galpy deprecated lists of potentials and will
    # drop them after 1.13.
    return dm + disc


# ----------------------------------------------------------------- agama --
def to_agama(params, baryons=True, lmax=8, rmin=None, rmax=None, **kw):
    """agama Potential, built as a Multipole over the density callable.

    agama's unit system is global, so this sets it to (Msun, kpc, km/s) --
    jeanie's convention -- which will affect any other agama object in the
    session. That is agama's design, not a choice made here, and it is the
    usual source of silently wrong results when mixing sources.
    """
    import agama
    agama.setUnits(mass=1, length=1, velocity=1)      # Msun, kpc, km/s
    r1 = float(params[2])
    rho = density_fn(params, **kw)

    def dens(xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        R = np.hypot(xyz[:, 0], xyz[:, 1])
        return np.asarray(rho(R, xyz[:, 2]), float)

    dm = agama.Potential(type="Multipole", density=dens,
                         symmetry="axisymmetric", lmax=lmax,
                         rmin=1e-3 * r1 if rmin is None else rmin,
                         rmax=60.0 * r1 if rmax is None else rmax)
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

    galax's API has moved more than galpy's, so this is written against the
    AbstractSinglePotential interface and raises with a clear pointer to
    `callables()` if that interface has changed -- wrapping four JAX
    functions by hand is less work than debugging a converter.
    """
    try:
        import equinox as eqx
        import jax.numpy as jnp
        import galax.potential as gp
        import unxt
    except ImportError as e:        # pragma: no cover - optional dependency
        raise ImportError(
            "to_galax needs galax (pip install galax). For any other "
            "framework use jeanie.export.callables(params), which returns "
            "plain JAX rho/M/g/Phi."
        ) from e

    Phi = potential_fn(params, **kw)
    Md, a, b = _mn(params)

    class JeanieHalo(gp.AbstractSinglePotential):
        """Spherically averaged jeanie halo, differentiable in params."""

        def _potential(self, q, t):                   # noqa: D401
            r = jnp.sqrt(jnp.sum(jnp.square(q), axis=-1))
            return Phi(r)

    try:
        dm = JeanieHalo(units=units)
    except Exception as e:          # pragma: no cover - API drift
        raise RuntimeError(
            "galax's potential interface has changed; build one yourself "
            "from jeanie.export.callables(params)['Phi'], which is the same "
            "traced JAX function this wrapper uses. Original error: %s" % e
        ) from e
    if not baryons:
        return dm
    disc = gp.MiyamotoNagaiPotential(m_tot=Md, a=a, b=b, units=units)
    return gp.CompositePotential(halo=dm, disc=disc)
