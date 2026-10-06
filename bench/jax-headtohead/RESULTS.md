# JAX prototype head-to-head

Three agents each built a JAX port of the 1D with-baryon solver. Their own
reports could not be compared: each used its own parameter sets, measured at a
different time under different machine load, and — the real confound — shipped
a different configuration. A defaults to `n_newton=4` and `UNROLL=4`; B and C
to `n_newton=3` and unroll=1, so part of "A is slow" was configuration rather
than architecture.

This runs all three at matched settings (`n_steps=200`, `n_gl=16`, `n_ramp=16`,
`n_newton=3`, unroll=1), on identical parameter sets, interleaved in one
process so load drift hits each equally, **pinned to one core**:

```sh
taskset -c 3 python bench/jax-headtohead/run.py
```

Pinning matters more than anything else here. Unpinned on a loaded box the
numba baseline inflates by ~1.8x, which is what produced the original and
wrong conclusion that JAX is slower than numba.

## Results

Pinned to one core, 15 reps x 12 parameter sets, three independent runs
agreeing to ~1%. Machine load 16-18 on 72 cores; the pinning isolates from it
(the numba baseline lands at 30.6-31.1 ms against a 29.3 ms clean reference).

| implementation | forward | vs numba | 2x6 Jacobian | ratio | vmap(64)/halo |
|----------------|---------|----------|--------------|-------|---------------|
| A pure `lax` + `custom_vjp` | 33.2 ms | 0.92x | 35.9 ms | 1.06x | 7.47 ms |
| **B `lax.custom_root`** | **23.2 ms** | **1.32x** | **26.6 ms** | 1.15x | **5.46 ms** |
| C literal port | 33.8 ms | 0.91x | 33.2 ms | 0.98x | 13.68 ms |
| numba | 30.7 ms | 1.00x | ~368 ms (12 solves) | — | 30.7 ms serial |

All three agree with numba to **2.5e-12** on r0 at matched settings, so
accuracy does not discriminate between them.

## Reading it

**B wins on every absolute measure** and the margin is larger than the
uncontrolled reports suggested: 1.43x faster than A and C on a forward solve,
and 2.5x better than C under vmap.

**C's Jacobian ratio of 0.98x is flattering, not good.** The ratio is against
its own slow forward solve; in absolute terms its Jacobian costs 33.2 ms
against B's 26.6 ms. Ratios to a self-baseline are the wrong comparison when
the baselines differ by 1.4x.

**A and C are statistically indistinguishable** (33.2 vs 33.8 ms) and both are
marginally *slower* than numba. Only B beats it.

**The gradient argument is the decisive one.** A full six-parameter Jacobian
costs B 26.6 ms. The same thing from numba is twelve central-difference solves,
~368 ms, and loses digits at small step sizes. That is 13.8x, and it is a
capability numba does not have at any price rather than a speed-up.

## Caveat

Everything here is CPU, single core, 1D with a Miyamoto-Nagai disc and a
spherical NFW outer halo. No GPU jaxlib is installed on this machine, and the
scalar-scan bottleneck that sets these numbers is exactly what a GPU would
treat differently, so none of this should be assumed to carry over.
