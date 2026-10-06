"""The exported density: the one object every downstream consumer wants.

galpy's MultipoleExpansionPotential.from_density, agama's Multipole and
galax/StreamSculptor all take a density callable (or a potential derived from
one), so this layer is what makes the halo usable as a potential at all. It
did not exist before: solve_log returns only [log r0, log sigma0] and
rk4_monopole's scan discarded the interior trajectory.
"""
import numpy as np
import pytest

from jeans.definitions import GN
from jeanie import jaxsolver as J
from jeanie.solver import solve_spherical

pytestmark = pytest.mark.skipif(not J.HAVE_JAX, reason="jax not installed")

if J.HAVE_JAX:
    import jax
    import jax.numpy as jnp

    from jeanie import jaxprofile as JP

P = [1e12, 10.0, 12.0, 6e10, 3.0, 0.28]        # M200, c, r1, Md, a, b


def _p():
    return jnp.asarray(P)


def test_trajectory_march_matches_the_endpoint_only_march():
    """Keeping the trajectory must not change the arithmetic.

    Same body, same order; the only difference is that the scan emits its
    carry. Anything but bit-identical endpoints means the two have drifted.
    """
    p = _p()
    M200, c, r1, Md, a, b = P
    lg = J.solve_log(p)
    r0, sig0sq = jnp.exp(lg[0]), jnp.exp(2.0 * lg[1])
    nodes = jnp.linspace(0.0, r1, 201)
    half = 0.5 * (nodes[:-1] + nodes[1:])
    h = nodes[1] - nodes[0]
    theta, w = (jnp.asarray(x) for x in J.gl_tables(J.N_GL_DEFAULT))
    s_n = J.source_from_grid(J.mn_grid(Md, a, b, nodes, theta), w, sig0sq, 1.0)
    s_h = J.source_from_grid(J.mn_grid(Md, a, b, half, theta), w, sig0sq, 1.0)

    phi1, eta1 = J.rk4_monopole(nodes, h, r0 ** 2, s_n, s_h)
    phi, eta = JP.rk4_monopole_traj(nodes, h, r0 ** 2, s_n, s_h)
    assert float(phi[-1]) == float(phi1)
    assert float(eta[-1]) == float(eta1)
    assert float(phi[0]) == 0.0 and float(eta[0]) == 0.0
    assert phi.shape == nodes.shape


def test_interior_density_matches_the_numpy_solver():
    p = _p()
    M200, c, r1, Md, a, b = P
    rho1, M1 = [float(v) for v in J.nfw_boundary(M200, c, r1)]
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    ref = solve_spherical(r1, rho1, M1, Phi_b=pb, trajectory=True)
    assert ref.success

    nodes, phi, eta, s_n, r0, sigma0 = JP.interior_profile(p)
    rho0 = float(sigma0 ** 2 / (4.0 * np.pi * GN * r0 ** 2))
    nd, ph = np.asarray(nodes), np.asarray(phi)
    for r in (1.0, 3.0, 6.0, 9.0, 11.5):
        got = rho0 * np.exp(-np.interp(r, nd, ph))
        assert got == pytest.approx(float(ref.rho(r)), rel=1e-12)


def test_angular_average_is_continuous_at_the_matching_radius():
    """The model matches <rho>_theta and M(<r1), not rho pointwise.

    So the test is that the ANGULAR AVERAGE joins the boundary value, and
    that the apparent step across r1 shrinks linearly with the offset, i.e.
    it is the profile's own slope rather than a seam.
    """
    p = _p()
    r1 = P[2]
    sph = JP.spherical_density_fn(p)
    rho1 = float(J.nfw_boundary(P[0], P[1], r1)[0])
    assert float(sph(r1 - 1e-6)) == pytest.approx(rho1, rel=1e-5)

    jumps = []
    for d in (1e-2, 1e-3, 1e-4):
        lo, hi = float(sph(r1 - d)), float(sph(r1 + d))
        jumps.append(abs(hi / lo - 1.0))
    for big, small in zip(jumps, jumps[1:]):
        assert small == pytest.approx(big / 10.0, rel=0.2), f"jumps {jumps}"


def test_the_interior_is_flattened_by_the_disc_and_the_exterior_is_not():
    """A sanity check on the angular structure, which is easy to get wrong.

    Inside r1 the DM feels the disc, so the density is higher in the plane
    than on the axis at the same radius. Outside, at q0 = 1, it is spherical.
    """
    p = _p()
    rho = JP.density_fn(p)
    for r in (2.0, 6.0, 11.0):
        assert float(rho(r, 0.0)) > float(rho(0.0, r)), f"not flattened at r={r}"
    for r in (15.0, 30.0):
        assert float(rho(r, 0.0)) == pytest.approx(float(rho(0.0, r)), rel=1e-12)


@pytest.mark.slow
def test_density_is_differentiable_in_every_halo_parameter():
    p = _p()

    def summary(q):
        f = JP.density_fn(q)
        return jnp.log(f(5.0, 1.0)) + jnp.log(f(0.0, 8.0))

    g = jax.grad(summary)(p)
    assert bool(jnp.all(jnp.isfinite(g)))
    for k in range(6):
        if abs(float(g[k])) < 1e-14:
            continue
        best = np.inf
        for h in (1e-4, 1e-5, 1e-6):
            d = jnp.zeros(6).at[k].set(h * abs(p[k]))
            fd = float((summary(p + d) - summary(p - d)) / (2 * h * abs(p[k])))
            best = min(best, abs(fd / float(g[k]) - 1.0))
        assert best < 1e-6, f"parameter {k}: best agreement with FD {best:.2e}"


def test_density_jits_and_vmaps():
    p = _p()
    f = jax.jit(lambda q, R, z: JP.density_fn(q)(R, z))
    assert np.isfinite(float(f(p, 5.0, 1.0)))
    V = jax.jit(jax.vmap(lambda m: JP.density_fn(p.at[0].set(m))(5.0, 1.0)))
    out = V(jnp.linspace(5e11, 2e12, 8))
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.all(jnp.diff(out) > 0)), "density should rise with M200"


def test_enclosed_mass_is_exact_at_the_matching_radius():
    """M(<r1) = M1 is a matching condition, so it holds to machine precision.

    That is also why the acceleration needs no patching at the join: M and
    rho are both continuous there, so g = -GM/r^2 is C1 across it.
    """
    p = _p()
    M200, c, r1 = P[0], P[1], P[2]
    M1 = float(J.nfw_boundary(M200, c, r1)[1])
    M = JP.mass_fn(p)
    assert float(M(r1)) == pytest.approx(M1, rel=1e-12)
    assert float(M(r1 + 1e-9)) == pytest.approx(float(M(r1 - 1e-9)), rel=1e-8)
    # and it must be monotone, which a bad interpolant quietly breaks
    rs = np.linspace(0.05, 3.0 * r1, 200)
    vals = np.array([float(M(r)) for r in rs])
    assert np.all(np.diff(vals) > 0), "enclosed mass is not monotone"


def test_radial_acceleration_is_finite_at_the_origin_and_points_inward():
    p = _p()
    g = JP.radial_acceleration_fn(p)
    assert float(g(0.0)) == 0.0
    for r in (0.1, 1.0, 5.0, 11.9, 12.1, 40.0):
        assert float(g(r)) < 0.0, f"not inward at r={r}"
    assert np.isfinite(float(g(1e-8)))


@pytest.mark.slow
def test_acceleration_gradients_match_finite_differences():
    """This is the quantity a stream likelihood differentiates, so pin it."""
    p = _p()
    f = lambda q: JP.radial_acceleration_fn(q)(3.7)
    g = jax.grad(f)(p)
    assert bool(jnp.all(jnp.isfinite(g)))
    for k in range(6):
        if abs(float(g[k])) < 1e-30:
            continue
        best = np.inf
        for h in (1e-4, 1e-5, 1e-6):
            d = jnp.zeros(6).at[k].set(h * abs(p[k]))
            fd = float((f(p + d) - f(p - d)) / (2 * h * abs(p[k])))
            best = min(best, abs(fd / float(g[k]) - 1.0))
        assert best < 1e-7, f"parameter {k}: {best:.2e}"


def test_hermite_beats_linear_on_smoothness():
    """Cubic Hermite is used instead of jnp.interp for a measured reason.

    Linear interpolation of M(<r) is C0, so its kinks land directly in
    d a / d r -- spurious stiffness to an adaptive integrator, noise to a
    gradient. The test compares the two on the same quantity rather than
    asserting an absolute threshold: the second difference of the
    acceleration straddling an interior node, where a linear interpolant
    kinks and a Hermite one does not.
    """
    p = _p()
    nodes, phi, eta, s_n, r0, sigma0 = JP.interior_profile(p)
    nd = np.asarray(nodes)
    h = float(nd[1] - nd[0])
    rho0 = float(sigma0 ** 2 / (4.0 * np.pi * GN * r0 ** 2))
    M_nodes = np.asarray(4.0 * np.pi / 3.0 * rho0 * nodes ** 3 * jnp.exp(-eta))

    g_herm = JP.radial_acceleration_fn(p)

    def g_lin(r):                       # the rejected alternative
        return -GN * float(np.interp(r, nd, M_nodes)) / r ** 2

    def curv(f, r_node, d):
        return abs(f(r_node - d) - 2.0 * f(r_node) + f(r_node + d))

    worse = []
    for i in (37, 57, 101, 151):
        rn, d = float(nd[i]), h / 8.0
        ch = curv(lambda r: float(g_herm(r)), rn, d)
        cl = curv(g_lin, rn, d)
        worse.append(cl / max(ch, 1e-300))
    # Measured ratios at these nodes are 9.0 to 22.1, so 5 is a floor with
    # margin rather than a guess. This is a LOCAL proxy: the effect that
    # motivated Hermite was on the gradient of an orbit loss, where the kinks
    # accumulate over thousands of steps and the gap was four orders of
    # magnitude (5e-5 against 1e-8), not one.
    assert min(worse) > 5.0, f"Hermite not clearly smoother: ratios {worse}"
