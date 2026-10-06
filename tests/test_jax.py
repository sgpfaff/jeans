"""The JAX backend, against the numba one.

Gradient correctness is checked against finite differences of the *numba*
solver, never against finite differences of the JAX solver itself. A prototype
that validated its gradients against its own forward solve passed easily and
was still wrong by factors of 10 to 7000 at non-converged points, because both
sides shared the same root. An independent implementation is the only check
worth running.
"""
import numpy as np
import pytest

from jeans.definitions import GN
from jeans.fast import jaxsolver as J
from jeans.fast.solver import solve_spherical

pytestmark = pytest.mark.skipif(not J.HAVE_JAX, reason="jax not installed")

if J.HAVE_JAX:
    import jax
    import jax.numpy as jnp

FIDUCIAL = [1e12, 10.0, 10.0, 6e10, 3.0, 0.28]
CASES = [
    FIDUCIAL,
    [5e12, 8.0, 20.0, 1e11, 4.0, 0.50],
    [3e11, 12.0, 6.0, 2e10, 2.0, 0.20],
    [1e13, 7.0, 30.0, 8e10, 5.0, 0.80],
]
# R > R_MAX: no solution exists, so the solver must say so rather than return
# something plausible.
NO_SOLUTION = [3e10, 14.0, 26.0, 6e10, 3.0, 0.28]


def numba_log_r0(p):
    M200, c, r1, Md, a, b = p
    rho1, M1 = [float(v) for v in J.nfw_boundary(M200, c, r1)]
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    out = solve_spherical(r1, rho1, M1, Phi_b=pb, verify=False)
    return np.log(out.r0) if out.success else np.nan


# ----------------------------------------------------------------- basics
def test_x64_is_on():
    """A silent drop to float32 would explain both fast timings and poor
    accuracy, so pin it rather than assume it."""
    assert jax.config.read("jax_enable_x64") is True
    out = J.solve_log(jnp.asarray(FIDUCIAL))
    assert out.dtype == jnp.float64


@pytest.mark.slow
@pytest.mark.parametrize("p", CASES)
def test_matches_numba(p):
    out = J.solve(np.asarray(p))
    assert out["ok"]
    assert out["r0"] == pytest.approx(np.exp(numba_log_r0(p)), rel=1e-9)


def test_nfw_boundary_matches_the_package():
    from jeans.classes import CDM_profile
    for M200, c, r1 in [(1e12, 10.0, 10.0), (1e13, 7.0, 30.0)]:
        o = CDM_profile(M200, c, q0=1.0)
        rho1, M1 = [float(v) for v in J.nfw_boundary(M200, c, r1)]
        assert rho1 == pytest.approx(o.rho_sph_avg(r1), rel=1e-12)
        assert M1 == pytest.approx(o.M_encl(r1), rel=1e-9)


# -------------------------------------------------------------- gradients
@pytest.mark.slow
def test_gradients_match_finite_differences_of_numba():
    """Independent check: the numba solver is a separate implementation."""
    worst = 0.0
    for p0 in CASES[:2]:
        p0 = np.asarray(p0)
        jac = np.asarray(jax.jacfwd(J.solve_log)(jnp.asarray(p0)))[0]
        for i in range(6):
            ad = jac[i] * p0[i]                     # chain to d/d log p
            s = 1e-5
            hi, lo = p0.copy(), p0.copy()
            hi[i] *= 1 + s
            lo[i] *= 1 - s
            fd = ((numba_log_r0(hi) - numba_log_r0(lo))
                  / (np.log1p(s) - np.log1p(-s)))
            worst = max(worst, abs(ad / fd - 1.0))
    assert worst < 1e-6, f"worst AD-vs-numba-FD disagreement {worst:.2e}"


def test_masked_failure_poisons_the_gradient():
    """The most dangerous failure mode this module exists to not have.

    jnp.where(bad, nan, out) returns NaN values but ZERO tangents, because the
    NaN branch is a constant. A sampler reads that as "insensitive to all six
    parameters" rather than "failed". Both behaviours are exercised here so
    the distinction cannot silently regress.
    """
    bad = jnp.asarray(NO_SOLUTION)

    val = np.asarray(J.solve_log(bad))
    grad = np.asarray(jax.jacfwd(J.solve_log)(bad))
    assert np.all(np.isnan(val)), "a non-existent solution must not return a number"
    assert np.all(np.isnan(grad)), "masked failure must poison its own gradient"

    def naive(q):
        xf, resid, _ = J._solve_core(q, 200, 16, 16, 3, 1, True)
        o = jnp.stack([0.5 * xf[0], 0.5 * xf[1]])
        return jnp.where(resid > J.RESID_TOL, jnp.nan, o)

    naive_grad = np.asarray(jax.jacfwd(naive)(bad))
    assert np.all(naive_grad == 0.0), (
        "the contrast case no longer demonstrates the bug; if jnp.where has "
        "started propagating NaN tangents, _poison may be unnecessary"
    )


def test_good_case_gradient_is_finite():
    g = np.asarray(jax.jacfwd(J.solve_log)(jnp.asarray(FIDUCIAL)))
    assert np.all(np.isfinite(g))
    assert np.any(np.abs(g) > 0)


# ------------------------------------------------------------ jit and vmap
def test_jit_does_not_retrace():
    """Recompiling per call would make every quoted timing meaningless."""
    f = jax.jit(J.solve_log)
    for p in CASES:
        f(jnp.asarray(p))
    assert f._cache_size() == 1


@pytest.mark.slow
def test_vmap_is_real_vectorisation():
    """A vmap that degrades to a loop gives right answers at no speed-up."""
    import time
    rng = np.random.default_rng(0)
    P = np.stack([10 ** rng.uniform(11.3, 12.9, 64), rng.uniform(6, 16, 64),
                  rng.uniform(4, 22, 64), 10 ** rng.uniform(9.5, 11.0, 64),
                  rng.uniform(1.5, 5.0, 64), rng.uniform(0.15, 0.9, 64)], axis=1)
    one = jax.jit(J.solve_log)
    many = jax.jit(jax.vmap(J.solve_log))
    one(jnp.asarray(P[0]))
    many(jnp.asarray(P))

    t0 = time.perf_counter()
    jax.block_until_ready(one(jnp.asarray(P[0])))
    t_one = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(many(jnp.asarray(P)))
    per = (time.perf_counter() - t0) / 64
    assert per < 0.5 * t_one, f"vmap gave {t_one/per:.1f}x, expected >2x"


@pytest.mark.slow
def test_vmap_values_match_scalar():
    rng = np.random.default_rng(2)
    P = np.stack([10 ** rng.uniform(11.5, 12.5, 8), rng.uniform(7, 13, 8),
                  rng.uniform(6, 18, 8), 10 ** rng.uniform(10.0, 10.8, 8),
                  rng.uniform(2.0, 4.0, 8), rng.uniform(0.2, 0.6, 8)], axis=1)
    batched = np.asarray(jax.vmap(J.solve_log)(jnp.asarray(P)))
    for i, p in enumerate(P):
        single = np.asarray(J.solve_log(jnp.asarray(p)))
        # XLA emits different vectorised code, so expect ulp-level agreement
        # rather than bit identity; a prototype claiming bit identity was wrong.
        assert batched[i] == pytest.approx(single, rel=1e-13)


def test_failures_stay_in_their_lane():
    """One bad halo must not poison the rest of the batch."""
    P = np.array(CASES[:3] + [NO_SOLUTION])
    out = np.asarray(jax.vmap(J.solve_log)(jnp.asarray(P)))
    assert np.all(np.isfinite(out[:3]))
    assert np.all(np.isnan(out[3]))


# ------------------------------------------------------------ the fold gate
@pytest.mark.slow
def test_schedule_independence_gate_runs_and_agrees_on_good_cases():
    for p in CASES:
        out = J.solve_verified(np.asarray(p))
        assert out["ok"], f"rejected a good solve: {out.get('reason')}"
        assert out["r0"] == pytest.approx(out["r0_dense"], rel=1e-6)


# --------------------------------------------------- differentiable boundary
@pytest.mark.slow
@pytest.mark.parametrize("M200,c,r1,q0", [
    (1e12, 10.0, 10.0, 1.0), (1e12, 10.0, 10.0, 0.8),
    (1e13, 7.0, 30.0, 0.7), (5e12, 8.0, 20.0, 0.9)])
def test_jax_boundary_matches_package(M200, c, r1, q0):
    """Closed-form squashed boundary against CDM_profile."""
    from jeans.classes import CDM_profile
    from jeans.fast import jaxouter as JO
    o = CDM_profile(M200, c, q0=q0)
    rho1, M1, JL = JO.boundary_data(M200, c, r1, q0=q0, L_list=(0, 2))
    pkJ = o.compute_potential_moments(r1, L_list=[0, 2], M_list=[0, 0])
    assert float(rho1) == pytest.approx(o.rho_sph_avg(r1), rel=1e-12)
    assert float(M1) == pytest.approx(o.M_encl(r1), rel=1e-8)
    assert float(JL[0]) == pytest.approx(pkJ[0], rel=1e-6)
    if pkJ[1]:
        assert float(JL[1]) == pytest.approx(pkJ[1], rel=1e-6)


def test_jax_boundary_reduces_to_nfw_boundary_when_spherical():
    from jeans.fast import jaxouter as JO
    a = JO.boundary_data(1e12, 10.0, 10.0, q0=1.0, L_list=(0,))
    b = J.nfw_boundary(1e12, 10.0, 10.0)
    assert float(a[0]) == pytest.approx(float(b[0]), rel=1e-14)
    assert float(a[1]) == pytest.approx(float(b[1]), rel=1e-12)


def test_boundary_gradients_reach_the_halo_parameters():
    """The point of the module: q0 was previously not differentiable at all."""
    from jeans.fast import jaxouter as JO
    f = lambda p: jnp.log(JO.enclosed_mass(10.0, p[0], p[1], p[2]))
    p0 = np.array([1e12, 10.0, 0.8])
    g = np.asarray(jax.grad(f)(jnp.asarray(p0)))
    assert np.all(np.isfinite(g))
    for i in range(3):
        s = 1e-5
        hi, lo = p0.copy(), p0.copy()
        hi[i] *= 1 + s
        lo[i] *= 1 - s
        fd = ((float(f(jnp.asarray(hi))) - float(f(jnp.asarray(lo))))
              / (np.log1p(s) - np.log1p(-s)))
        assert g[i] * p0[i] == pytest.approx(fd, rel=1e-6)


@pytest.mark.slow
def test_full_2d_chain_is_differentiable_including_q0():
    """boundary -> 2D solve, one traced expression, gradients in all 7 params.

    The q0 derivative is compared in absolute terms: d log r0 / d log q0 is
    ~1e-6 here because the rho1 and M1 shifts very nearly cancel, so a
    relative comparison is dominated by the finite difference's own noise.
    """
    from jeans.classes import CDM_profile
    from jeans.fast import jaxouter as JO, jaxsolver2d as J2
    from jeans.fast.outer import boundary_data as np_boundary
    from jeans.fast.solver2d import solve_axisymmetric

    Ls = (0, 2)

    def disc(Md, a, b):
        return lambda r, th: -GN * Md / np.sqrt(
            r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)

    def chain(q):
        rho1, M1, JL = JO.boundary_data(q[0], q[1], q[2], q0=q[6], L_list=Ls)
        return J2.solve_log_2d(q[:6], L_list=Ls, J_L=JL, boundary=(rho1, M1))

    p = np.array([1e12, 10.0, 10.0, 6e10, 3.0, 0.28, 0.8])
    o = CDM_profile(p[0], p[1], q0=p[6], Phi_b=disc(*p[3:6]))
    rho1, M1, JL = np_boundary(o, p[2], L_list=Ls)
    ref = solve_axisymmetric(p[2], rho1, M1, J_L=JL, L_list=Ls, Phi_b=disc(*p[3:6]))

    got = float(np.exp(np.asarray(chain(jnp.asarray(p)))[0]))
    assert got == pytest.approx(ref.r0, rel=1e-6)

    jac = np.asarray(jax.jacfwd(chain)(jnp.asarray(p)))[0]
    assert np.all(np.isfinite(jac))

    def numba_log_r0(pp):
        oo = CDM_profile(pp[0], pp[1], q0=pp[6], Phi_b=disc(*pp[3:6]))
        rr, mm, jj = np_boundary(oo, pp[2], L_list=Ls)
        r = solve_axisymmetric(pp[2], rr, mm, J_L=jj, L_list=Ls, Phi_b=disc(*pp[3:6]))
        return np.log(r.r0) if r.success else np.nan

    for i in range(7):
        s = 1e-5
        hi, lo = p.copy(), p.copy()
        hi[i] *= 1 + s
        lo[i] *= 1 - s
        fd = (numba_log_r0(hi) - numba_log_r0(lo)) / (np.log1p(s) - np.log1p(-s))
        ad = jac[i] * p[i]
        assert abs(ad - fd) < 1e-6 + 1e-5 * abs(fd), f"param {i}: {ad} vs {fd}"


# ------------------------------------------------- Einasto and contraction
# Tolerances here are deliberately asymmetric. Against an exact reference
# (closed form, or adaptive quadrature with Richardson extrapolation) the
# requirement is machine precision. Against the PACKAGE the requirement is
# loose, because the package is the approximate side: it splines the
# spherically averaged baryon potential on 100 points and differentiates the
# spline, and recovers rho from M by np.gradient on a 1000-point grid.
# Measured against adaptive quadrature at r = 2, 5, 10, 20, 40 kpc, this
# implementation is accurate to 2.7e-11 .. 1.3e-7 while the package's M_b is
# accurate to 4.3e-6 .. 8.9e-5.

def _mn_jnp(Md, a, b):
    return lambda r, th: -GN * Md / jnp.sqrt(
        r ** 2 * jnp.sin(th) ** 2
        + (a + jnp.sqrt(b ** 2 + r ** 2 * jnp.cos(th) ** 2)) ** 2)


def _mn_np(Md, a, b):
    return lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)


def test_einasto_matches_the_package_to_machine_precision():
    from jeans.cdm import M_Einasto, rho_Einasto
    from jeans.fast import jaxouter as JO
    M200, c, alpha = 1e12, 10.0, 0.18
    rs = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
    rho_m2, r_m2, r200 = JO.einasto_params(M200, c, alpha)
    pk_rho = rho_Einasto(M200, c, alpha, mass_concentration=True)
    pk_M = M_Einasto(M200, c, alpha, mass_concentration=True)
    for r in rs:
        assert float(JO.einasto_rho(r, rho_m2, r_m2, alpha)) == pytest.approx(
            float(pk_rho(r)), rel=1e-12)
        assert float(JO.einasto_mass(r, M200, c, alpha)) == pytest.approx(
            float(pk_M(r)), rel=1e-12)
    # the normalisation is the defining property, so check it separately
    assert float(JO.einasto_mass(float(r200), M200, c, alpha)) == pytest.approx(
        M200, rel=1e-12)


def test_baryon_mass_beats_the_package_against_exact_quadrature():
    """M_b = r^2/G d<Phi_b>/dr, by autodiff rather than by splining."""
    from scipy.integrate import quad
    from jeans.tools import compute_Mb
    from jeans.fast import jaxouter as JO
    Md, a, b = 5e10, 2.5, 0.4
    pj, pn = _mn_jnp(Md, a, b), _mn_np(Md, a, b)
    r200 = float(JO.einasto_params(1e12, 10.0, 0.18)[2])
    pk = compute_Mb(pn, 1e-10 * r200, 1e2 * r200)

    def avg(r):
        return 0.5 * quad(lambda th: pn(r, th) * np.sin(th), 0, np.pi,
                          epsabs=1e-15, epsrel=1e-13, limit=400)[0]

    for r in (2.0, 5.0, 10.0, 20.0):
        prev = ref = None
        for h in (r * 1e-2, r * 5e-3, r * 2.5e-3):
            d = (avg(r + h) - avg(r - h)) / (2 * h)
            if prev is not None:
                ref = r ** 2 / GN * (d + (d - prev) / 3.0)   # Richardson
            prev = d
        mine = abs(float(JO.mb_spherical(pj, r)) / ref - 1.0)
        theirs = abs(float(pk(r)) / ref - 1.0)
        assert mine < 1e-6, f"r={r}: {mine:.2e}"
        assert mine < theirs, f"r={r}: mine {mine:.2e} not better than {theirs:.2e}"


def test_cautun_does_not_reduce_to_the_uncontracted_profile():
    """At zero baryon mass the Cautun factor is 0.861, not 1.

    M_CDM there is the total-matter profile and the prescription removes the
    cosmological baryon fraction, so AC-on and AC-off are different models of
    the same halo rather than a small correction to one. Worth pinning,
    because it is the first thing that looks like a bug and is not.
    """
    from jeans.fast import jaxouter as JO
    M_cdm = lambda r: jnp.asarray(1.0)
    zero = lambda r: jnp.asarray(0.0)
    f = float(JO.ac_mass(10.0, M_cdm, zero, "Cautun"))
    assert f == pytest.approx(0.45 + 0.38 * 1.16 ** 0.53, rel=1e-12)
    assert f == pytest.approx(0.8611, abs=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("presc", ["Cautun", "Gnedin"])
def test_adiabatic_contraction_matches_the_package(presc):
    from jeans.cdm import AC_profiles
    from jeans.tools import compute_Mb
    from jeans.fast import jaxouter as JO
    M200, c, Md, a, b = 1e12, 10.0, 5e10, 2.5, 0.4
    pj, pn = _mn_jnp(Md, a, b), _mn_np(Md, a, b)
    r200 = float(JO.einasto_params(M200, c, 0.18)[2])
    pk_Mb = compute_Mb(pn, 1e-10 * r200, 1e2 * r200)
    _, pk_M = AC_profiles(M200, c, pk_Mb, AC_prescription=presc)
    rho_s, r_s = JO.nfw_params(M200, c)
    M_cdm = lambda r: JO.nfw_mass(r, rho_s, r_s)
    M_b = lambda r: JO.mb_spherical(pj, r)
    for r in (2.0, 5.0, 10.0, 20.0, 40.0):
        mine = float(JO.ac_mass(r, M_cdm, M_b, presc, r200))
        # loose: the package's M_b spline feeds straight into this
        assert mine == pytest.approx(float(pk_M(r)), rel=2e-3)


@pytest.mark.slow
def test_contracted_boundary_gradients_reach_every_halo_parameter():
    """Gradients must survive Einasto, contraction and the squashing together."""
    from jeans.fast import jaxouter as JO
    pj = _mn_jnp(5e10, 2.5, 0.4)

    def f(p):
        rho1, M1, J_L = JO.boundary_data(
            p[0], p[1], 10.0, q0=p[2], L_list=(0, 2),
            halo_type="Einasto", alpha=p[3],
            AC_prescription="Cautun", Phi_b=pj)
        return jnp.array([jnp.log(rho1), jnp.log(M1), J_L[1]])

    p0 = jnp.array([1e12, 10.0, 0.8, 0.18])
    jac = jax.jacfwd(f)(p0)
    assert bool(jnp.all(jnp.isfinite(jac))), "non-finite Jacobian"
    for k in range(4):
        best = np.inf
        for h in (1e-4, 1e-5, 1e-6):
            d = jnp.zeros(4).at[k].set(h * abs(p0[k]))
            fd = (f(p0 + d) - f(p0 - d)) / (2 * h * abs(p0[k]))
            best = min(best, float(jnp.max(jnp.abs(fd - jac[:, k])
                                           / (jnp.abs(jac[:, k]) + 1e-30))))
        assert best < 1e-6, f"parameter {k}: best agreement with FD {best:.2e}"
