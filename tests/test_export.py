import numpy as np
import pytest

from jeanie.export import callables, to_galpy
from jeanie.jaxprofile import potential_fn, radial_acceleration_fn

PARAMS = np.array([1.0e12, 10.0, 15.0, 6.0e10, 3.0, 0.28])
RO, VO = 8.0, 220.0
GN = 4.302e-6


def test_potential_is_consistent_with_the_acceleration():
    """dPhi/dr must equal -g_r. The two are built by different routes, so
    this catches an inconsistent interpolation rather than an algebra slip.
    Linear interpolation of Phi left this at 1.4e-2; Hermite with dPhi/du
    taken exactly from M(r) brings it to ~1e-5."""
    import jax
    Phi = potential_fn(PARAMS)
    g = radial_acceleration_fn(PARAMS)
    r = np.geomspace(0.5, 200.0, 16)
    num = np.array([float(jax.grad(Phi)(float(x))) for x in r])
    ref = np.array([float(-g(float(x))) for x in r])
    assert np.median(np.abs(num / ref - 1)) < 1e-4
    assert np.max(np.abs(num / ref - 1)) < 1e-3


def test_potential_stays_differentiable_in_the_halo_parameters():
    """The point of the JAX export: d(orbit)/d(halo params) without finite
    differences. A Phi that is only differentiable in r is useless for that."""
    import jax
    gr = jax.grad(lambda q: potential_fn(q)(20.0))(PARAMS)
    assert np.all(np.isfinite(gr))
    assert abs(float(gr[2])) > 0.0          # responds to r1


def test_galpy_spherical_round_trips_the_force():
    """galpy's own force evaluation must reproduce jeanie's g_r.

    This is the unit conversion, which is the thing that goes silently wrong:
    a potential with the wrong scaling still integrates orbits, just the
    wrong ones."""
    galpy = pytest.importorskip("galpy.potential")
    pot = to_galpy(PARAMS, ro=RO, vo=VO, kind="spherical", baryons=False)
    g = callables(PARAMS)["g"]
    F0 = VO ** 2 / RO
    for r in (1.0, 5.0, 20.0, 60.0, 150.0):
        got = float(galpy.evaluateRforces(pot, r / RO, 0.0,
                                          use_physical=False)) * F0
        assert abs(got / float(g(r)) - 1.0) < 1e-5


def test_galpy_baryon_amplitude_is_exact():
    """The Miyamoto-Nagai amplitude conversion into galpy internal units."""
    galpy = pytest.importorskip("galpy.potential")
    pot = to_galpy(PARAMS, ro=RO, vo=VO, baryons=True)
    assert len(pot) == 2            # combined potentials still len() as 2
    Md, a, b = PARAMS[3], PARAMS[4], PARAMS[5]
    R = 20.0
    exact = -GN * Md * R / (R ** 2 + (a + b) ** 2) ** 1.5
    got = float(galpy.evaluateRforces(pot[1], R / RO, 0.0,
                                      use_physical=False)) * VO ** 2 / RO
    assert abs(got / exact - 1.0) < 1e-10


def test_baryons_actually_change_the_force():
    """`baryons=False` is a silent trap -- an orbit outside r1 then has zero
    gradient w.r.t. Md, a and b. Assert the two differ."""
    galpy = pytest.importorskip("galpy.potential")
    F0 = VO ** 2 / RO
    dm = to_galpy(PARAMS, ro=RO, vo=VO, baryons=False)
    tot = to_galpy(PARAMS, ro=RO, vo=VO, baryons=True)
    a = float(galpy.evaluateRforces(dm, 20.0 / RO, 0.0, use_physical=False)) * F0
    b = float(galpy.evaluateRforces(tot, 20.0 / RO, 0.0, use_physical=False)) * F0
    assert abs(b) > 1.2 * abs(a)


def test_callables_are_the_framework_agnostic_escape_hatch():
    c = callables(PARAMS)
    assert set(c) == {"rho", "M", "g", "Phi"}
    assert float(c["M"](20.0)) > 0.0
    assert float(c["rho"](8.0, 0.0)) > float(c["rho"](8.0, 6.0))   # flattened
