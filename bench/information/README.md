# Where is the information about the cross-section?

**Status: the pipeline works, the numbers are not converged.**

Ground truth is generated **by the model**, so there is no misspecification at
all. Any failure to recover is information content or sampler behaviour, not a
physics mismatch -- which is the thing the EAGLE run (`../eagle/`) could not
separate, and why its 1.8x bias was ambiguous.

One mock galaxy-scale halo (`M200 = 1e12`, `c = 10`, `q0 = 0.7`, a
Milky-Way disc, `sigma/m = 0.5`), fitted under three likelihoods.

| likelihood | recovered `sigma/m` | truth | fractional width |
|---|---|---|---|
| density `rho(r)` only | 0.289 +0.268 -0.138 | 0.5 | 1.404 |
| shape `q(r)` only | 0.527 +0.504 -0.240 | 0.5 | 1.412 |
| **both** | **0.406 +0.134 -0.109** | 0.5 | **0.598** |

> **These numbers are NOT converged. Read the warning below before using
> them.**

The integrated autocorrelation time is tau ~ 12-15 against 150 post-burn
steps, so **n_eff ~ 10-12 effective samples** in every chain. emcee's own
diagnostic says so: "the chain is shorter than 50 times the integrated
autocorrelation time". At ten effective samples the 16/84 percentiles are
noise, and the following do NOT follow from this run:

* that density and shape carry *equal* information. 1.404 against 1.412 is
  indistinguishable here; the true values could be 1.0 and 2.0.
* that the combination is worth *2.35x*. The factor is not measured.

What does survive: all three recover `sigma/m` within a factor of two of the
truth, and the joint constraint is visibly tighter than either single
observable (see `corner_information.png`, where the joint contour is enclosed
by both). The ordering is probably real; nothing quantitative is.

The shape-only posterior is also broad and structured, with 0.6% of samples
near the upper `sigma/m` prior edge, so part of it may be prior-informed
rather than data-informed.

Getting this right needs roughly 3000 steps rather than 300, which is ten
hours per fit at the current cost -- so the `r1` bracket caching below is a
prerequisite, not an optimisation.

## Caveats

300 steps at acceptance ~0.5, one noise realisation, and the disc is fixed and
known. The *widths* are the quantity of interest and they are the more robust
part, but none of this is a converged production run. `q(r)` also carries
strong information about the baryons -- the core is more flattened than the
outer halo (`q` runs 0.36 to 0.87 against `q0 = 0.7`) because the dark matter
follows the disc-dominated potential -- so with a free disc the shape
constraint would partly go into the baryons instead.

## Running it

```
PYTHONPATH=../../src python run_info.py rho  300
PYTHONPATH=../../src python run_info.py q    300
PYTHONPATH=../../src python run_info.py both 300
python plot_info.py
```

About 30 minutes each on 8 cores. The cost is dominated by the `sigma/m -> r1`
inversion, which re-brackets from scratch every call; caching the previous
`r1` would cut it to a handful of solves and is the first thing to do before
any production run.

**Use a spawn context, not fork.** JAX is multithreaded and
`multiprocessing.Pool` forks by default on Linux; the combination deadlocks
silently -- no error, no output, just a hang until the timeout. It is a race,
so it sometimes works, which is worse.
