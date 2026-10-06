# Can jeanie recover a known cross-section from a simulation?

EAGLE-50 knows the answer: `SIDM1b` was run at `sigma/m = 1 cm^2/g`, `CDMb` at
zero. Feed each halo's own baryons in as `Phi_b`, fit its dark-matter density
profile, and see what comes back. If the model cannot recover a known
cross-section from a simulation it will not recover one from a stream track,
and this needs no stream code at all.

Five halos ship with the repo (`examples/data/EAGLE-50/`); the full 250 are on
[Zenodo](https://zenodo.org/records/16331984). Note these are **clusters**,
`M200` from 1.8e13 to 1.6e14 Msun — the data is in Mpc and `simulations.py`
converts. So this validates the model, at cluster scale; it does not directly
probe the galaxy-scale regime a stellar stream lives in.

## Results

| halo | model | core | recovered `sigma/m` | truth | ratio | scatter |
|---|---|---|---|---|---|---|
| 1 | SIDM1b | 19 kpc | 1.68 +0.12 -0.11 | 1 | 1.7x | 0.067 dex |
| 4 | SIDM1b | 11 kpc | 1.96 +0.11 -0.16 | 1 | 2.0x | 0.050 dex |
| 2 | SIDM1b | none | 0.068 +0.077 -0.041 | 1 | - | 0.176 dex |
| 2 | CDMb | n/a | 0.050 +-0.020 | 0 | - | 0.223 dex |

Three things to take from this.

**It discriminates.** A halo with a core returns ~1.8; a collisionless halo
returns 0.05. A factor of thirty.

**It is biased by a consistent ~1.8x on the halos that have signal.** The two
agree to 1.5 sigma on formal errors that plainly understate the systematics,
so "consistent" is as strong as two halos support. The likely cause is a
convention rather than a bug:

    sigma/m = Nm / (rho1 * vrel * t_age) * conversion,   vrel = 4 sigma0/sqrt(pi)

`Nm = 1` scatter per particle at `r1` and `t_age = 10 Gyr` are definitions of
the matching radius, not measurements. There is no guarantee the core that
forms in a simulation corresponds to `Nm` exactly 1; an effective `Nm ~ 0.55`
would remove the offset. Calibrating that against the full 250-halo set is
the obvious next step, and would turn this from a bias into a measurement.

**`SIDM1b` halo 2 returning 0.068 is probably correct, not a failure.** Its
density profile sits within 6% of its CDM counterpart and never falls below
half of it: there is no core to find. The simulation ran at `sigma/m = 1`, but
at cluster mass with a dominant BCG the contraction has erased the signature.
That is a caution for the stream programme -- `sigma/m = 1` does not guarantee
an observable core, and whether it does depends on the baryons.

The intrinsic scatter is a usable goodness-of-fit diagnostic: 0.05-0.07 dex
where the model applies, 0.18-0.22 where it does not. For a stacked analysis
that is how you decide which halos to include.

## What had to be got right

Four things went wrong first, each caught only by checking something else.

1. **Fit the concentration outside the core.** Determining `c` on
   [0.15, 1.0] `R200` includes radii the core still affects, which biases `c`
   low, which biases `r1`, which biases `sigma/m`. On halo 2 that alone moved
   the answer from 0.995 to 0.125.
2. **Sample `sigma/m`, not `r1`.** `r1` is defined by the scattering
   condition, so fitting it freely adds a degree of freedom the physics does
   not have -- and the corner plot showed that freedom being spent trading
   `r1` against `c`. Sampling `sigma/m` and solving for `r1` (a bracketed
   bisection, since `sigma/m` is monotone in `r1`) removed a 10x bias.
3. **Start every walker feasible.** The bisection cannot bracket for every
   `(M200, c, sigma/m)`. Sixteen of thirty-two walkers began at `-inf`, never
   moved, and their parameter values then contaminated the posterior summary
   -- dragging a quoted median from 0.050 to 0.170 and making a failed
   recovery look consistent with the truth.
4. **Include adiabatic contraction.** Without it `M200` came out 0.11 dex
   below the simulation's; with the Cautun prescription it recovers to 0.03
   dex. It did NOT fix `sigma/m`, which is how the remaining bias was
   localised to the convention rather than to the outer halo.

## Files

| file | what it does |
|---|---|
| `point_estimate.py` | grid scan in `r1`, no sampling. Fast, and shows the `c`-`r1` sensitivity directly |
| `mcmc_nfw.py` | joint posterior, uncontracted NFW outer halo |
| `mcmc_contracted.py` | the same with Cautun contraction. This is the one to use |
| `corner_plot.py` | corner plots from the saved chains |
| `summary_figure.py` | the four runs on one axis -> `eagle_recovery_summary.png` |

```
PYTHONPATH=../../src python mcmc_contracted.py SIDM1b 1 800    # ~8 min on 26 cores
PYTHONPATH=../../src python mcmc_contracted.py CDMb   2 800    # the control
```

Both scripts pre-tabulate the simulation's `Phi_b`, which is a vectorised
spline over a `solve_ivp` dense output that `_Problem` calls ~400 times per
construction.
