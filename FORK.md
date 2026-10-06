# About this fork

A working fork of [dark-physics/jeans](https://github.com/dark-physics/jeans)
(Bautista, Robertson, Sagunski, Smith-Orlik & Tulin,
[arXiv:2511.10765](https://arxiv.org/abs/2511.10765)) with two goals: fix
defects found while validating the package, and add reduced solvers fast and
smooth enough for Bayesian inference from extragalactic stellar streams.

No new physics. The model is theirs; this is an implementation of the same
equations by a different numerical route, plus bug fixes.

Background and measurements: **[Fast Differentiable SIDM Halos](https://claude.ai/artifact/MeerL1uT1moD3YUd4jXik8)**.

## What is here

### Defect fixes (`src/jeans/`)

Six defects, each pinned by a regression test in `tests/test_defects.py` written
to fail on upstream first. All ten tests pass here and all ten fail against
upstream.

| ID | Where | Effect |
|----|-------|--------|
| D1 | `classes.py` | `sph_sym_flag` never recomputed by `update()`, so every output accessor reported a spherically symmetric halo and all quadrupole observables were silently zeroed. Hits the no-baryon case and all spheroidal hosts. |
| D2 | `cdm.py`, `classes.py` | `AC_profiles` hardcoded NFW: Einasto+AC was byte-identical to NFW+AC. Honouring `halo_type` exposed a second defect underneath, where Einasto's finite total mass made the density reconstruction return NaN at every radius. |
| D3 | `classes.py` | Three broken accessors on the `profile` wrapper. |
| D4 | `tools.py` | Guard referenced an undefined name, raising `NameError` instead of its message. |
| D5 | `classes.py` | `M_list=[0,0,0]` rejected although every entry is zero. |
| D6 | `gen.py` | `jeans.spherical` used `m0 = exp(-<Phi_b>)` while `jeans.isothermal` used the rigorous `m0 = <exp(-Phi_b)>`. They differ by a Jensen gap of up to 1.8% in `r0`, so `jeans.spherical` was not the `L=[0]` limit of `jeans.isothermal`. |

D2's NaN and D6's Jensen gap both change published numbers, not just error paths.

### Reduced solvers (`src/jeans/fast/`)

```python
from jeans.classes import CDM_profile
from jeans.fast.outer import boundary_data
from jeans.fast.solver2d import solve_axisymmetric

halo = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=my_disc)
rho1, M1, J_L = boundary_data(halo, 10.0, L_list=(0, 2))
res = solve_axisymmetric(10.0, rho1, M1, J_L=J_L, L_list=(0, 2), Phi_b=my_disc)
res.r0, res.sigma0, res.phi_at(2, 9.9)
```

| module | contents |
|--------|----------|
| `universal.py` | The parameter-free interior solution and the no-baryon matching, which is an interpolation rather than a root-find. Carries the exact existence criterion `R < 1.2615266`. |
| `kernels.py` | Numba RK4 for the monopole and the linearised multipole system. Fixed step count, which is what keeps gradients smooth. |
| `quadrature.py` | Fixed-node Gauss-Legendre in `cos(theta)` with cached harmonics, replacing adaptive `scipy.quad`. |
| `solver.py` | The spherical solver: interpolation without baryons, a bracketed monotone inversion with them. |
| `branch.py` | Branch selection. The matching problem is multi-valued; this decides which root is physical, and supplies the exact existence criterion with baryons. |
| `solver2d.py` | The axisymmetric solver: nonlinear monopole alternating with a linear-response shape solve. |
| `outer.py` | Outer-halo boundary data on fixed nodes, replacing two nested adaptive quadratures. |
| `jaxsolver.py` | The differentiable backend: same algorithm, exact gradients by implicit differentiation, jit and vmap. |
| `jaxouter.py` | Closed-form differentiable boundary data, so gradients reach `M200`, `c` and `q0`. |

Measured against the package, all at matched accuracy:

| case | package | reduced | speed-up | agreement |
|------|---------|---------|----------|-----------|
| 1D, no baryons | 0.31 s | 0.03 ms | ~10,000x | 8e-5 |
| 1D + Miyamoto-Nagai disc | 2.0 s | 37 ms | 59x | 2.0e-4 |
| 1D + Hernquist spheroid | 0.30 s | 16 ms | 20x | 6.0e-5 |
| 2D `L=[0,2]`, q0=0.8, no baryons | 6.2 s | 6.4 ms | 960x | 7.8e-5 |
| 2D `L=[0,2]` + disc | 15.1 s | 39 ms | 390x | 2.0e-4 |
| 2D `L=[0,2,4]` + disc | 30.4 s | 57 ms | 530x | 2.1e-4 |

Outer-halo boundary data, which became the bottleneck once the interior solve
was reduced:

| quantity | package | reduced | speed-up | agreement |
|----------|---------|---------|----------|-----------|
| `J_L`, q0=0.8 + disc | 1029 ms | 8.4 ms | 123x | 3.0e-7 |
| `M(<r1)`, q0=0.8 | 48 ms | 7.3 ms | 6.6x | 5.7e-11 |
| full chain, q0=0.8 | 1413 ms | 14.6 ms | **97x** | - |

Both were nested adaptive quadrature. `compute_potential_moments` ran an
adaptive `solve_ivp` in r whose integrand was an adaptive `quad` in theta,
repeated over widening radial shells until the tail converged; under
`t = r1/r` the infinite tail maps to `t -> 0` where the integrand is finite,
so one fixed Gauss-Legendre rule covers the whole range. `M_encl` rebuilt a
300-point spline per call with an angular average at every point; the
substitution `r = r1 v^2` turns its `r^-1` central cusp into a smooth `v^3`,
so no inner cutoff is needed either.

Those are monopole errors. The shape `phi_L` carries an extra error quadratic
in the halo's own multipoles, because the L>0 sector is linearised about the
spherical background: `phi_2` agrees to 4e-5 with a disc and 3e-3 at q0=0.8
without one. `phi_4` is a different matter -- at the parameters tested
`phi_2^2 = 2.6e-3` is the same size as `phi_4 = 2.5e-3`, so the dropped
quadratic term is the *leading* contribution to L=4 rather than a correction
to it, and L=4 comes out 3.3% low. Treat L=4 as indicative until a
second-order or block-tridiagonal solve replaces the linearisation.

The residual is the package's own error, not the solver's:
`test_package_converges_toward_the_fast_solution` asserts that refining the
package's radial grid closes the gap as `O(N^-2)` with no floor.

## Branch selection

The matching problem has more than one root, and past a fold the residual stops
being a correctness test: a measured case returns an `r0` 201 times too small
with a residual of 2.1e-13. This is the classical isothermal spiral -- `R` is
the reciprocal Milne homology variable, the folds are the classical locus
`u_M + v_M = 3`, and they space out as `exp(2*pi/sqrt(7)) = 10.75`. For the
dark-matter-only case the multi-valuedness is already in the literature
([Robertson et al. 2021](https://arxiv.org/abs/2009.07844), Appendix A).

What is here is the criterion with baryons, and a solver that cannot reach the
wrong branch. The reduced map `(u1, Lam) -> (R, mu)` is explicit, so folds are
exactly where its Jacobian determinant vanishes, and two monotonicity results
make the inversion fully bracketable. `solve_spherical` now defaults to that
bracketed inversion.

Measured on 4000 draws from a stream-inference prior and 4000 from a
fold-enriched prior, against ground truth from exhaustive, path-independent
root enumeration:

| | wrong answers | caught by the certificate | caught by schedule independence | median cost |
|---|---|---|---|---|
| plain Newton | 3.8% / 10.0% | - | - | 0.006 s |
| 16-stage ramp | 0.125% / 0.300% | 5/5, 12/12 | 5/5, 11/12 | 0.044 s |
| ramp + schedule screen | - | 0 false positives | 1 and 20 false positives | 0.128 s |
| **bracketed (default)** | cannot mis-branch | - | - | **0.018 s** |

Two things worth knowing beyond the error rate.

**The existence criterion is exact and cheap.** A solution exists iff
`1/3 < R < R_fold(mu; shape)`, which generalises the no-baryon `R < R_MAX`.
The lower bound is the `u1 -> 0` limit of the map and is new here; it is what
rules out heavily baryon-dominated configurations.

**Continuation loses solutions it cannot reach.** Above `R_MAX` the ramp's
first stage is the baryon-free problem, which has no solution, so the path does
not exist at its own starting point. Over 2889 fold-prior draws that do have a
solution, the ramp reached 6.2% of the `R > R_MAX` cases and `n_ramp=128` did
no better. These are silent coverage losses rather than wrong answers, and the
bracketed solver recovers them.

The `u1 < 22.5441` fast path is not a theorem -- a Plummer sphere at
`b/r1 = 3` puts the first fold at 22.33819 -- but every counterexample found
requires `R > R_MAX`, so it is applied only when `R < R_MAX` and the exact
determinant scan runs otherwise. Full discussion in the `branch.py` docstring.

## Status

Defect fixes, the 1D and 2D reduced solvers, the fast outer halo, the
benchmark harness and the differentiable 1D JAX backend are done. Still to
come: the JAX outer halo beyond NFW, the 2D path in JAX, and a second-order or
block-tridiagonal treatment of L=4.

### The differentiable backend

```python
import jax, jax.numpy as jnp
from jeans.fast.jaxsolver import solve_log      # (M200, c, r1, Md, a, b)

p = jnp.array([1e12, 10.0, 10.0, 6e10, 3.0, 0.28])
jax.jacfwd(solve_log)(p)                        # exact 2x6 Jacobian
jax.vmap(solve_log)(P)                          # a whole ensemble
```

Pinned to one core, matched settings, against numba:

| | JAX | numba |
|---|-----|-------|
| forward solve | 23.7 ms | 30.8 ms |
| full 2x6 Jacobian | 26.5 ms (1.12x forward) | ~370 ms (12 FD solves) |
| vmap(64), solve only | 5.4 ms/halo | 30.8 ms/halo serial |
| vmap(64), solve + Jacobian | **6.0 ms/halo** | not available |

Gradients agree with central finite differences of the numba solver -- a
genuinely separate implementation -- to 3.5e-8 over twelve log-derivatives.
Agreement on r0 is 1.7e-12.

Three failure modes it deliberately does not have, each found by an
adversarial verifier on a prototype that did:

* **A masked failure poisons its own gradient.** `jnp.where(bad, nan, out)`
  returns NaN values but *zero* tangents, which a sampler reads as "this halo
  is insensitive to all six parameters" rather than "this halo failed". That
  is the one error that silently corrupts a posterior. Both behaviours are
  pinned by a test.
* **`lax.custom_root` never checks it found a root.** At a non-converged point
  it returns a clean-looking Jacobian, measured wrong by factors of 10 to 7000.
  Masking is on by default rather than assuming convergence.
* **Past a fold the residual is not a correctness test.** `solve_verified`
  re-solves at twice the ramp density and requires agreement.

Measurements and methodology: `bench/README.md`, and the
[benchmark report](https://claude.ai/artifact/2yZYkZpa6fyeNns5qzMJsw).

Nothing here has been offered upstream yet. The defect fixes are useful on their
own and are the natural first contribution.

## Running the tests

The suite splits into unit tests, which touch no reference profile, and
comparisons against the package, which dominate the runtime because a single
package build costs 0.3-30 s.

```sh
PYTHONPATH=src python -m pytest -m "not slow" -q   # 8 tests, 0.9 s
PYTHONPATH=src python -m pytest -q                 # 63 tests, 2.3 min
```

Package references are memoised for the session in `tests/conftest.py`; only
package output is cached, never the fast solvers', so a test cannot pass by
comparing a cached value against itself.

Confirm the defect tests reproduce against upstream:

```sh
PYTHONPATH=/path/to/upstream/jeans/src python -m pytest tests/test_defects.py -q
```
