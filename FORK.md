# About this fork

A working fork of [dark-physics/jeans](https://github.com/dark-physics/jeans)
(Tulin & Smith-Orlik, [arXiv:2511.10765](https://arxiv.org/abs/2511.10765)) with
two goals: fix defects found while validating the package, and add reduced
solvers fast and smooth enough for Bayesian inference from extragalactic
stellar streams.

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
from jeans.fast.solver import solve_spherical

outer = CDM_profile(1e12, 10.0, q0=1.0, Phi_b=my_disc)
res = solve_spherical(10.0, outer.rho_sph(10.0), outer.M_encl(10.0), Phi_b=my_disc)
res.r0, res.sigma0
```

| module | contents |
|--------|----------|
| `universal.py` | The parameter-free interior solution and the no-baryon matching, which is an interpolation rather than a root-find. Carries the exact existence criterion `R < 1.2615266`. |
| `kernels.py` | Numba RK4 for the monopole and the linearised multipole system. Fixed step count, which is what keeps gradients smooth. |
| `quadrature.py` | Fixed-node Gauss-Legendre in `cos(theta)` with cached harmonics, replacing adaptive `scipy.quad`. |
| `solver.py` | The spherical solver: interpolation without baryons, continuation in the baryon amplitude with them. |
| `solver2d.py` | The axisymmetric solver: nonlinear monopole alternating with a linear-response shape solve. |

Measured against the package, all at matched accuracy:

| case | package | reduced | speed-up | agreement |
|------|---------|---------|----------|-----------|
| 1D, no baryons | 0.31 s | 0.03 ms | ~10,000x | 8e-5 |
| 1D + Miyamoto-Nagai disc | 2.0 s | 37 ms | 59x | 2.0e-4 |
| 1D + Hernquist spheroid | 0.30 s | 16 ms | 20x | 6.0e-5 |
| 2D `L=[0,2]`, q0=0.8, no baryons | 6.2 s | 6.4 ms | 960x | 7.8e-5 |
| 2D `L=[0,2]` + disc | 15.1 s | 39 ms | 390x | 2.0e-4 |
| 2D `L=[0,2,4]` + disc | 30.4 s | 57 ms | 530x | 2.1e-4 |

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

## Status

Stage 0 (defect fixes) and Stage 1 (1D and 2D reduced solvers) are done. Still
to come: a JAX backend with implicit-function gradients, a second-order or
block-tridiagonal treatment of L=4, and the benchmark harness.

Nothing here has been offered upstream yet. The defect fixes are useful on their
own and are the natural first contribution.

## Running the tests

The suite splits into unit tests, which touch no reference profile, and
comparisons against the package, which dominate the runtime because a single
package build costs 0.3-30 s.

```sh
PYTHONPATH=src python -m pytest -m "not slow" -q   # 8 tests, 0.9 s
PYTHONPATH=src python -m pytest -q                 # 47 tests, 2.6 min
```

Package references are memoised for the session in `tests/conftest.py`; only
package output is cached, never the fast solvers', so a test cannot pass by
comparing a cached value against itself.

Confirm the defect tests reproduce against upstream:

```sh
PYTHONPATH=/path/to/upstream/jeans/src python -m pytest tests/test_defects.py -q
```
