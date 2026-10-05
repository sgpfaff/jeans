"""Timing and accuracy measurement with enough rigour to quote the results.

The numbers this replaces were single `time.time()` calls around one
invocation on a shared machine. That is not good enough to publish: it cannot
distinguish a real speed-up from a quiet moment on the box, and for the numba
paths it risks folding JIT compilation into the measurement.

What this does instead:

* discards warmup runs explicitly, so compilation is never timed;
* repeats and reports the median with an interquartile range, not a single
  sample, so the spread is visible;
* reports the minimum as well, which for a deterministic computation on a busy
  machine is the closest thing to the true cost;
* records the machine state (load average, CPU count, library versions) beside
  the numbers, so a later run can be compared honestly.

Package builds cost 0.3-30 s, so they get few repeats; the reduced solvers get
many. Both report their own spread, and the speed-up carries the combined
uncertainty rather than being quoted as a bare ratio.
"""
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["Timing", "measure", "environment", "speedup_interval"]


@dataclass
class Timing:
    median_s: float
    min_s: float
    q1_s: float
    q3_s: float
    n: int
    warmup: int
    # Every sample, not just the summary. Keeping these is what lets a figure
    # show the actual scatter: on a log axis spanning seven decades a 2%
    # interquartile range is sub-pixel, so error bars alone silently read as
    # "no uncertainty was measured".
    samples_s: list = field(default_factory=list)

    @property
    def iqr_frac(self):
        """Interquartile spread as a fraction of the median: the noise level."""
        return (self.q3_s - self.q1_s) / self.median_s if self.median_s else float("nan")


def measure(fn, repeats=7, warmup=1, max_seconds=120.0):
    """Time fn() with warmup discarded. Returns (Timing, last_result).

    Stops early once max_seconds of wall clock has been spent, so a 30 s
    package build does not turn into a five-minute measurement; the Timing
    records how many repeats actually ran.
    """
    for _ in range(warmup):
        fn()
    samples = []
    t_start = time.perf_counter()
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > max_seconds:
            break
    samples.sort()
    n = len(samples)
    q1 = samples[max(0, int(0.25 * (n - 1)))]
    q3 = samples[min(n - 1, int(0.75 * (n - 1)))]
    return Timing(statistics.median(samples), samples[0], q1, q3, n, warmup,
                  samples_s=samples), out


def speedup_interval(slow: Timing, fast: Timing):
    """Median speed-up plus a conservative interval from the two IQRs.

    Pairing slow-q1 with fast-q3 and vice versa gives a wider interval than
    propagating in quadrature. That is deliberate: these are wall-clock
    measurements on a shared machine, where the errors are neither Gaussian
    nor independent, and overstating the precision of a speed-up is exactly
    the failure mode this module exists to prevent.
    """
    return {
        "median": slow.median_s / fast.median_s,
        "lo": slow.q1_s / fast.q3_s,
        "hi": slow.q3_s / fast.q1_s,
        "best_case": slow.min_s / fast.min_s,
    }


def environment():
    """Machine and library state, recorded beside every result."""
    import scipy
    try:
        import numba
        numba_v = numba.__version__
    except ImportError:
        numba_v = None
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_v,
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": [load1, load5, load15],
    }


def rel_err(got, want):
    return float(abs(got / want - 1.0)) if want else float("nan")


def density_error(fast_rho, pkg_rho, r_grid, theta_grid):
    """max |rho_fast/rho_pkg - 1| over a radius-by-angle grid."""
    worst = 0.0
    for r in r_grid:
        for th in theta_grid:
            a, b = fast_rho(r, th), pkg_rho(r, th)
            if b:
                worst = max(worst, abs(a / b - 1.0))
    return worst


def save(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_encode)
    return path


def _encode(o):
    if isinstance(o, Timing):
        return asdict(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
