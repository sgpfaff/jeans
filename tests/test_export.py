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


# ---------------------------------------------------------- the multipole --
@pytest.fixture(scope="module")
def mp():
    from jeanie.export import multipole
    return multipole(PARAMS, lmax=8)


def test_the_multipole_monopole_is_jeanies_own_enclosed_mass(mp):
    """Two routes to M(<r): the ODE solution integrated directly, and the
    l = 0 harmonic of the density it produced. They are independent, so
    agreement at 1e-5 checks both."""
    M = callables(PARAMS)["M"]
    for r in (1.0, 5.0, 15.0, 60.0, 200.0):
        assert abs(float(mp.enclosed_mass(r)) / float(M(r)) - 1) < 1e-4


def test_the_multipole_reproduces_the_density_it_was_built_from(mp):
    """On the reference grid (see to_galpy) this is 2.5e-4 median and 2.9e-2
    worst at lmax = 8, falling to 1.2e-4 / 3.5e-3 at lmax = 24. The worst
    case is the midplane inside r1, where a 0.28 kpc disc pinches the
    isothermal solution into an angular feature. Compare the monopole, which
    kind='spherical' keeps: 2.6e-2 median. Compare SCF: 40%."""
    rho = callables(PARAMS)["rho"]
    pts = [(1., 0.), (5., 5.), (10., 4.), (15.5, 0.), (20., 10.),
           (40., 0.), (150., 0.)]
    err = [abs(float(mp.density(R, z)) / float(rho(R, z)) - 1)
           for R, z in pts]
    assert np.median(err) < 2e-3
    assert max(err) < 2e-2


def test_galpy_multipole_round_trips_through_galpys_own_evaluation():
    """Only the unit conversion is under test here, so it should be exact:
    galpy is handed R and z in ro and must give back vo^2 and vo^2/ro."""
    galpy = pytest.importorskip("galpy.potential")
    from jeanie.export import to_galpy
    pot = to_galpy(PARAMS, ro=RO, vo=VO, kind="multipole", baryons=False)
    exp = pot._expansion
    D0 = (VO ** 2 * RO / GN) / RO ** 3
    for R, z in [(1., 0.), (5., 2.), (15., 0.), (60., 30.), (200., 0.)]:
        p = float(galpy.evaluatePotentials(pot, R / RO, z / RO,
                                           use_physical=False)) * VO ** 2
        f = float(galpy.evaluateRforces(pot, R / RO, z / RO,
                                        use_physical=False)) * VO ** 2 / RO
        d = float(galpy.evaluateDensities(pot, R / RO, z / RO,
                                          use_physical=False)) * D0
        assert abs(p / float(exp.potential(R, z)) - 1) < 1e-12
        assert abs(f / float(exp.forces(R, z)[0]) - 1) < 1e-12
        assert abs(d / float(exp.density(R, z)) - 1) < 1e-12


def test_galpy_multipole_keeps_the_flattening_spherical_throws_away():
    """The reason the multipole exists. The reference halo runs from q = 0.62
    at 3 kpc to q = 0.93 at 14 kpc -- a shape gradient set by the disc -- and
    kind='spherical' reports q = 1 everywhere."""
    pytest.importorskip("galpy.potential")
    from scipy.optimize import brentq
    from jeanie.export import to_galpy
    exp = to_galpy(PARAMS, ro=RO, vo=VO, kind="multipole",
                   baryons=False)._expansion
    q = []
    for r in (3.0, 14.0):
        target = float(exp.density(r, 0.0))
        q.append(brentq(lambda zz: float(exp.density(0.0, zz)) - target,
                        0.1 * r, 3.0 * r) / r)
    assert q[0] < 0.75                 # genuinely flattened at 3 kpc
    assert q[1] > q[0] + 0.15          # and the gradient is there
    assert q[1] < 1.0


def test_galpy_multipole_conserves_energy_on_an_orbit():
    """A potential whose forces are not the exact derivative of its own
    interpolant leaks energy, and the leak looks like physics."""
    pytest.importorskip("galpy.potential")
    from galpy.orbit import Orbit
    from jeanie.export import to_galpy
    pot = to_galpy(PARAMS, ro=RO, vo=VO, kind="multipole", baryons=True)
    o = Orbit([30. / RO, 0.0, 180. / VO, 2. / RO, 20. / VO, 0.0],
              ro=RO, vo=VO)
    ts = np.linspace(0.0, 2.0, 401)
    o.integrate(ts, pot, method="dop853")
    E = o.E(ts, pot=pot, use_physical=False)
    assert np.ptp(E) / abs(np.mean(E)) < 1e-9


def test_an_unknown_kind_is_refused():
    pytest.importorskip("galpy.potential")
    from jeanie.export import to_galpy
    with pytest.raises(ValueError, match="multipole"):
        to_galpy(PARAMS, kind="octopole")


# ----------------------------------------------------------------- agama --
@pytest.fixture(scope="module")
def ag():
    pytest.importorskip("agama")
    from jeanie.export import to_agama
    return to_agama(PARAMS, baryons=False)


def _xyz(pts):
    return np.array([[R, 0.0, z] for R, z in pts])


AG_PTS = [(1., 0.), (5., 5.), (10., 4.), (15.5, 0.), (20., 10.),
          (40., 0.), (150., 0.)]


def test_agama_reproduces_the_input_density(ag):
    """agama's Multipole is built from the same callable but on its own
    radial grid. At its defaults it is about 5x behind jeanie's expansion in
    the median -- 1.3e-3 against 2.5e-4 on the reference grid -- and it
    catches up at lmax=12 with gridsize=40. Both floors are the midplane
    inside r1. The tolerance here is set for the defaults."""
    rho = callables(PARAMS)["rho"]
    got = ag.density(_xyz(AG_PTS))
    ref = np.array([float(rho(R, z)) for R, z in AG_PTS])
    assert np.median(np.abs(got / ref - 1)) < 5e-3
    assert np.max(np.abs(got / ref - 1)) < 3e-2


def test_agama_and_jeanies_multipole_agree_on_the_force(ag, mp):
    """Two independent implementations of the same expansion. This is the
    only check here that is not against a closed form, and it is the one
    that would catch a shared misunderstanding of the model rather than an
    arithmetic slip."""
    f = ag.force(_xyz(AG_PTS))
    for (R, z), fa in zip(AG_PTS, f):
        FR, Fz = mp.forces(R, z)
        assert abs(float(FR) / fa[0] - 1) < 3e-3
        if abs(z) > 0:
            assert abs(float(Fz) / fa[2] - 1) < 3e-3


def test_agama_and_jeanie_agree_on_the_potential(ag, mp):
    """To 4e-3, and the disagreement is an offset of order 85 (km/s)^2 out
    of 1e5, not a scaling. Two parts to it: agama applies the CODATA G to
    jeanie's densities (2.5e-4, uniform) and the two codes continue the mass
    beyond rmax differently (the rest, which grows with radius). Neither
    shows up in the forces, which agree to 3e-3."""
    rel, absd = [], []
    for R, z in AG_PTS:
        a = float(mp.potential(R, z))
        b = ag.potential([[R, 0.0, z]])[0]
        rel.append(abs(a / b - 1))
        absd.append(abs(a - b))
    assert max(rel) < 4e-3
    assert max(absd) < 150.0


G_CODATA = 4.30091727e-6        # what agama and galax both use


def test_agama_baryon_component_is_the_analytic_disc_in_agamas_G():
    """Exact once the right G is used -- which is the point. agama puts
    CODATA on everything it builds, so the disc is 2.5e-4 shallower than the
    same Miyamoto-Nagai in jeanie's units. Asserting it against CODATA to
    1e-6, rather than against 4.302e-6 to 3e-4, is what makes this test fail
    if the amplitude conversion ever drifts for a real reason."""
    pytest.importorskip("agama")
    from jeanie.export import to_agama
    tot = to_agama(PARAMS, baryons=True)
    dm = to_agama(PARAMS, baryons=False)
    Md, a, b = PARAMS[3], PARAMS[4], PARAMS[5]
    R = 20.0
    disc = (tot.potential([[R, 0.0, 0.0]])[0]
            - dm.potential([[R, 0.0, 0.0]])[0])
    assert abs(disc / (-G_CODATA * Md / np.sqrt(R ** 2 + (a + b) ** 2))
               - 1) < 1e-6
    assert abs(disc / (-GN * Md / np.sqrt(R ** 2 + (a + b) ** 2))
               - 1) == pytest.approx(1.0 - G_CODATA / GN, rel=1e-3)


# ----------------------------------------------------------------- galax --
KMS2_IN_KPC_MYR2 = 1.0459401725324528e-06


def test_galax_potential_is_in_galax_units_not_jeanies():
    """Regression, and the reason this module exists. galax's "galactic"
    system measures speed in kpc/Myr; jeanie works in km/s. Handing galax
    jeanie's number unconverted left the halo 9.56e5 times too shallow beside
    a galax MiyamotoNagaiPotential built from the same params -- and the
    composite still integrated orbits, around the disc."""
    pytest.importorskip("galax.potential")
    import jax.numpy as jnp
    from jeanie.export import to_galax
    dm = to_galax(PARAMS, baryons=False)
    Phi = callables(PARAMS)["Phi"]
    for r in (1.0, 20.0, 150.0):
        got = float(dm.potential(jnp.array([r, 0.0, 0.0]), 0.0))
        assert abs(got / (float(Phi(r)) * KMS2_IN_KPC_MYR2) - 1) < 1e-10


def test_galax_halo_and_disc_end_up_in_the_same_units_AND_the_same_G():
    """galax's own MiyamotoNagaiPotential would be 2.5e-4 shallower than the
    halo beside it, because galax uses CODATA and the halo was solved with
    4.302e-6. A mismatch between two components of one potential is worse
    than a uniform offset from CODATA, so the disc is jeanie's own analytic
    Miyamoto-Nagai here. Exact, with nothing relabelled."""
    pytest.importorskip("galax.potential")
    import jax.numpy as jnp
    from jeanie.export import to_galax
    tot = to_galax(PARAMS, baryons=True)
    dm = to_galax(PARAMS, baryons=False)
    Md, a, b = PARAMS[3], PARAMS[4], PARAMS[5]
    for r in (5.0, 20.0, 60.0):
        q = jnp.array([r, 0.0, 0.0])
        disc = float(tot.potential(q, 0.0)) - float(dm.potential(q, 0.0))
        exact = (-GN * Md / np.sqrt(r ** 2 + (a + b) ** 2)
                 * KMS2_IN_KPC_MYR2)
        assert abs(disc / exact - 1) < 1e-10


def test_galax_acceleration_is_minus_grad_of_its_own_potential():
    pytest.importorskip("galax.potential")
    import jax.numpy as jnp
    from jeanie.export import to_galax
    tot = to_galax(PARAMS, baryons=True)
    h = 1e-4
    for r in (5.0, 20.0, 60.0):
        acc = np.asarray(tot.acceleration(jnp.array([r, 0.0, 0.0]), 0.0))[0]
        fd = -(float(tot.potential(jnp.array([r + h, 0.0, 0.0]), 0.0))
               - float(tot.potential(jnp.array([r - h, 0.0, 0.0]), 0.0))) \
            / (2 * h)
        assert abs(acc / fd - 1) < 1e-6


def test_galax_stays_differentiable_in_the_halo_parameters():
    """The whole reason to export to galax rather than galpy. float() on a
    tracer raises, so the disc parameters are left unconverted on this path;
    forcing them would silently rule this out."""
    pytest.importorskip("galax.potential")
    import jax
    import jax.numpy as jnp
    from jeanie.export import to_galax

    def f(p):
        return to_galax(p, baryons=True).potential(
            jnp.array([20.0, 0.0, 5.0]), 0.0)

    gr = np.asarray(jax.grad(f)(jnp.asarray(PARAMS)))
    assert np.all(np.isfinite(gr))
    assert abs(gr[2]) > 0.0            # responds to r1
    assert abs(gr[3]) > 0.0            # and to the disc mass
