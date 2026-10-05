# Benchmarks

Reproducible measurements of the reduced solvers against the reference package.

```sh
PYTHONPATH=src python bench/run.py           # all sections -> bench/results/*.json
PYTHONPATH=src python bench/run.py cases     # just one
PYTHONPATH=src python bench/plot.py          # figures -> bench/figs/
```

Results are versioned JSON, so a regression shows up as a diff rather than as
a number nobody remembers. Every figure is drawn from those files, so a plot
can never disagree with the recorded measurement.

## Methodology

The numbers this replaces were single `time.time()` calls around one
invocation on a shared machine. That cannot distinguish a real speed-up from a
quiet moment on the box, and for the numba paths it risks timing JIT
compilation. So `harness.py`:

* discards warmup runs explicitly;
* reports the median with an interquartile range, not one sample;
* also reports the minimum, which on a busy machine is the closest thing to
  the true cost of a deterministic computation;
* records load average, CPU count and library versions beside the numbers.

Speed-ups carry a conservative interval formed by pairing the slow q1 with the
fast q3 and vice versa, rather than propagating in quadrature. These are
wall-clock measurements on a shared machine; the errors are neither Gaussian
nor independent, and overstating a speed-up is the failure mode this exists to
prevent. `bench5_noise.png` shows why it matters: package spread is usually
under 4% but reached 27% on one case.

## Two speed-ups, and they differ a lot

`solver_only` compares the interior solve, which is what the reduction
replaces. `end_to_end` adds building the outer CDM halo, which both paths need
and neither reduction touches.

They are far apart — 19-29,826x against 1.5-28x — and the reason is worth
stating plainly: **once the interior solve is reduced, the outer halo becomes
the bottleneck.** For the 1D no-baryon case the interior solve costs 0.009 ms
and the outer halo about 160 ms. Quoting only `solver_only` would describe the
method honestly and the user's experience not at all.

## Sections

| section | what it measures |
|---------|------------------|
| `cases` | cost and accuracy across eight configurations |
| `cost` | accuracy against wall clock as the step count varies |
| `gridref` | whether the gap to the package closes as `O(N^-2)` |
| `ladder` | finite-difference stability against step size |
| `batch` | throughput over a 64-halo ensemble |
