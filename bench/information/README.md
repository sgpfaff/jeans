# Where is the information about the cross-section?

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

**Density and shape carry essentially equal information** -- 1.404 against
1.412. That was not what I expected; one seemed likely to dominate.

**Combining them is worth 2.35x, not sqrt(2).** Two independent measurements
of equal precision would give 1.41. Getting 2.35 means they break a
degeneracy rather than averaging: each one's `sigma/m` uncertainty is largely
shared with `(M200, c)`, and the two share it differently. The nuisance
parameters show it -- density alone pulls `c` to 12.1 and shape alone to 8.1
against a truth of 10, while together they land on 10.8.

**Shape alone is the least biased**, 0.527 against 0.5.

This is the gap between the two published approaches. X-Stream
([2508.02666](https://arxiv.org/abs/2508.02666)) constrains the radial
profile; Curve-Away ([2609.40057](https://arxiv.org/abs/2609.40057))
constrains the shape. Neither can combine them, because neither has a model
predicting both from a single `sigma/m`. This says the combination is worth
substantially more than either alone -- which is the argument for the 2D
machinery.

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
