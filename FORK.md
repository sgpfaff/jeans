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

Measured against the package, all at matched accuracy:

| case | package | reduced | speed-up | agreement |
|------|---------|---------|----------|-----------|
| 1D, no baryons | 0.31 s | 0.03 ms | ~10,000x | 8e-5 |
| 1D + Miyamoto-Nagai disc | 2.0 s | 37 ms | 59x | 2.0e-4 |
| 1D + Hernquist spheroid | 0.30 s | 16 ms | 20x | 6.0e-5 |

The residual is the package's own error, not the solver's:
`test_package_converges_toward_the_fast_solution` asserts that refining the
package's radial grid closes the gap as `O(N^-2)` with no floor.

## Status

Stage 0 (defect fixes) and part of Stage 1 (1D reduced solver) are done. Still
to come: the 2D linear-response solver, a JAX backend with implicit-function
gradients, and the benchmark harness. The 2D kernel exists but is not yet
wired up or validated.

Nothing here has been offered upstream yet. The defect fixes are useful on their
own and are the natural first contribution.

## Running the tests

```sh
PYTHONPATH=src python -m pytest tests/ -q
```

Confirm the defect tests reproduce against upstream:

```sh
PYTHONPATH=/path/to/upstream/jeans/src python -m pytest tests/test_defects.py -q
```
