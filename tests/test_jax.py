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
