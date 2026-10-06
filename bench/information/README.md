# Where is the information about the cross-section?

**Status: the pipeline works; the numbers are preliminary and carry
bootstrap errors.**

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

> **Preliminary: the chains are short. Bootstrap errors are quoted; read the
> note below.**

Bootstrapping over walkers gives the sampling error on each width:

| | width | bootstrap sigma |
|---|---|---|
| density | 1.404 | 0.173 |
| shape | 1.412 | 0.121 |
| both | 0.598 | 0.054 |

**Density and shape carry comparable information** -- the widths agree to
about 15%. They are not demonstrably *equal*; an earlier version of this file
said so, which claimed a precision the data does not have.

**Combining them is worth 2.35 +- 0.36**, so sqrt(2) = 1.41, the value for two
independent measurements of equal precision, is disfavoured at roughly
2.6 sigma. The two observables therefore appear to break a degeneracy rather
than simply average, which is visible in the nuisance parameters: density
alone pulls `c` to 12.1 and shape alone to 8.1 against a truth of 10, while
together they land on 10.8.

**Shape alone is the least biased**, 0.527 against 0.5.

Health of the chains: the integrated autocorrelation time is tau ~ 12-15
against 150 post-burn steps, so the run is about 11.5 tau long against
emcee's guidance of 50 tau, with roughly 277 effective samples
(n_steps x n_walkers / tau). The bootstrap above may therefore understate
the error, since it resamples walkers that are correlated along their own
length. Nothing here should be quoted without the longer run.

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
