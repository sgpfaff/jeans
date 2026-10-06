# About this fork

A working fork of [dark-physics/jeans](https://github.com/dark-physics/jeans)
(Bautista, Robertson, Sagunski, Smith-Orlik & Tulin,
[arXiv:2511.10765](https://arxiv.org/abs/2511.10765)) with two goals: fix
defects found while validating the package, and add reduced solvers fast and
smooth enough for Bayesian inference from extragalactic stellar streams.

The model is theirs. This solves the same equations by a different numerical
route, fixes defects found while validating them, and adds two results about
the model that the implementation made visible: an exact existence criterion
with baryons, and a branch criterion for the matching problem, which is
multi-valued. Neither changes the physics; both change which answer you get
out of it.

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
| `jaxouter.py` | Closed-form differentiable boundary data: NFW or Einasto, optional adiabatic contraction, constant flattening. Gradients reach `M200`, `c`, `q0` and the Einasto index. |

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

| | wrong answers | caught by the certificate | caught by schedule independence |
|---|---|---|---|
| plain Newton | 3.8% / 10.0% | - | - |
| 16-stage ramp | 0.125% / 0.300% | 5/5, 12/12 | 5/5, 11/12 |
| ramp + schedule screen | - | 0 false positives | 1 and 20 false positives |
| **bracketed (default)** | cannot mis-branch | - | - |

Cost is measured separately, in one process pinned to one core, over 40 cases
solvable by every method, 9 interleaved repetitions (`bench/branch/timing.py`).
The multiprocess campaign above is not a usable timer: three methods doing 1x,
16x and 32x the solve work reported the same wall time to three digits.

| method | ms per solve | IQR |
|---|---|---|
| plain Newton | 3.44 | 0.12 |
| 16-stage ramp | 25.74 | 1.64 |
| ramp + certificate | 28.34 | 1.55 |
| ramp + schedule screen | 70.90 | 3.85 |
| **bracketed (default)** | **13.04** | 1.06 |

So the default is 2.0x faster than the ramp alone and 5.4x faster than the
ramp with the screen it replaces. The certificate costs 2.6 ms on top of a
ramp; the screen costs 45 ms, because it is a second solve.

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

## The outer halo, differentiably

`jaxouter.py` reproduces `CDM_profile`'s dispatch -- NFW or Einasto, optionally
contracted by the Cautun or Gnedin prescription, squashed by a constant `q0` --
as one traced expression. Three things the package computes numerically are
exact here, and each was costing smoothness rather than only speed:

| quantity | package | here |
|---|---|---|
| baryon enclosed mass `M_b(r)` | spline of the averaged potential on 100 points, differentiated | fixed-node angular average, radial derivative by autodiff |
| contracted density | `np.gradient` of `log M` on a 1000-point grid, then splined | `M'(r)/4 pi r^2` by autodiff of the closed-form mass |
| Gnedin initial radius | `find_ri` iterated to a tolerance, raising after 1000 steps | the same fixed point under `lax.custom_root` |

Measured against adaptive quadrature with Richardson extrapolation, `M_b` here
is accurate to 2.7e-11 at 2 kpc and 1.3e-7 at 40 kpc, where the package's
spline is accurate to 8.9e-5 and 4.3e-6. Einasto matches the package's own
closed forms to 5e-15 in density and 7e-15 in mass, with `M(r200)/M200 = 1` to
twelve digits. Gradients in `M200`, `c`, `q0` and the Einasto index `alpha`
agree with finite differences to about 1e-10, on a plateau flat across three
step sizes. `jax.scipy.special.gammainc` turns out to be differentiable in its
*order* as well as its argument, which is what makes `alpha` inferable rather
than merely settable.

Adiabatic contraction is not a small correction. Over a realistic halo+disc
prior at `r1` = 5-30 kpc, switching it on moves the dark-matter density by a
median **-11.5%** (Cautun) or **-14.0%** (Gnedin), spanning -18.5% to +14%,
and the sign depends on the configuration. Part of that is bookkeeping rather
than contraction: at zero baryon mass the Cautun factor is
`0.45 + 0.38 * 1.16**0.53 = 0.861`, not 1, because `M_CDM` there is the
total-matter profile and the prescription removes the cosmological baryon
fraction. AC-on and AC-off are different models of the same halo.

The number that matters for inference is not the boundary shift but what it
does to the solved core. Over 120 halos drawn from the same prior, solved
with AC off and on:

| | core radius `r0` | dispersion `sigma0` | central density `rho0` |
|---|---|---|---|
| Cautun (n=110) | **-13.9%** (p10 -35.9%, max 56.5%) | -4.0% | +24.9% (p90 +148%, max 452%) |
| Gnedin (n=112) | **-10.7%** (p10 -39.8%, max 60.1%) | -6.8% | +7.7% (p90 +172%, max 646%) |

`r0` is essentially what constrains the cross-section, so a tens-of-percent
median shift with a 60% tail is not a correction to a fit, it is a different
fit. Running without contraction is a modelling choice that has to be stated,
not a simplification.

One more reason the package path was not viable here: a single contracted
boundary evaluation costs it 90.7 s against 9.2 s for this one including JAX
tracing. At 90 s a chain of 1e5 samples is about 100 days, so adiabatic
contraction was unusable for inference before it was undifferentiable.

## How far the linear shape sector actually goes

`solver2d` linearises the L>0 sector about the spherical background. Its
docstring used to quote "max |phi_2| ... below 0.272 over 1080
configurations" as the validated range, without saying what exceeding it
costs. Two things were wrong with that. 35% of a realistic prior exceeds it --
over 230 converged configurations with `q0` from 0.4 to 0.95 the median is
0.20 and the maximum 0.59, almost entirely driven by flat halos. And
exceeding it turns out to cost nothing worth worrying about.

Measured against the package's own relaxation, which builds its source as the
exact angular integral of `exp(-phi_b - sum_L phi_L Z_L)` and so assumes no
linearity, over 150 configurations spanning `max abs(phi_2)` from 0.02 to 0.94:

| `max abs(phi_2)` | n | median err `r0` | median err `phi_2` |
|---|---|---|---|
| [0.00, 0.05) | 6 | 1.6e-4 | 1.1e-3 |
| [0.15, 0.20) | 13 | 1.8e-4 | 1.7e-3 |
| [0.30, 0.40) | 17 | 2.1e-4 | 3.0e-3 |
| [0.50, 0.70) | 20 | 2.0e-4 | 5.0e-3 |
| [0.70, 0.94] | 7 | 2.0e-4 | 7.3e-3 |

There is no knee. A resolution ladder says why the `r0` column is flat:
refining both sides together (`r_grid` = `n_steps` = 200, 400, 800, 1600)
drives the `r0` difference 1.61e-4 to 3.82e-5 to 7.68e-6 to **6.52e-8**, while
the `phi_2` difference sits at 4.95e-4, 5.05e-4, 5.06e-4, 5.07e-4. So the flat
2e-4 was the comparison's own discretisation, and **the linearisation error in
the solved `r0` and `sigma0` is below 1e-7** -- the monopole is simply
indifferent to the shape amplitude (correlation of the `r0` error with
`abs(phi_2)`: -0.03).

The shape carries the only real error, and it is predictable:
`err/abs(phi_2)` is constant at 9.3e-3 (p10 4.4e-3, p90 1.4e-2), the signature
of the dropped quadratic term. At `abs(phi_2)` = 0.94 -- `q0` = 0.25, flatter
than any real halo -- the shape is still right to 0.96%.

`Result2D` therefore reports `max_psi` and a calibrated `shape_rel_error`
rather than a pass/fail flag: at the point a flag would have fired the answer
is good to 1%, and warning there would only teach people to ignore warnings.
A hard refusal remains at `max_psi > 1.2`, which exists to avoid
extrapolating a measured curve into unmeasured territory rather than to guard
against any realistic halo. `bench/shape/` reproduces all of it.

`L=4` is a separate matter, and structural. Its relative error is 1.7-6.9%
(median 3.3%), does not shrink as the halo rounds, and does not move under
refinement -- 3.20e-2, 3.22e-2, 3.23e-2 at `n_steps` = 200, 400, 800, and
identical at `n_outer` = 4 and 12 -- while `r0` over the same ladder falls
2.36e-4 to 2.48e-5. So it is the diagonal approximation: each `L` is marched
independently, dropping the 2-4 coupling. Including the off-diagonal block
would fix it, which is the same block-tridiagonal solve `L=6` needs.

That is a relative error on a small quantity rather than a reason to drop
`L=4`. `phi_4` is 10-20x smaller than `phi_2`, so the absolute contributions
are about equally accurate (1.1e-3 against 1.7e-3 at `q0` = 0.5). Including
`L=4` still improves the density; quoting `phi_4` as a precise amplitude does
not.

This also settles a question left open about 2D branch selection. The concern
was that a fold found in 2D might be the linearisation failing rather than
real physics, which would make a 2D certificate unassertable. The reported
failures sit at `max_psi` >= 1.4, i.e. `abs(phi_2)` ~ 2.3 -- more than twice
anything reachable even at `q0` = 0.25. That regime is not a grey area, it is
unphysical.

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
