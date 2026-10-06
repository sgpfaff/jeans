"""Like-for-like timing. Every method goes through the same public entry point.

The earlier harness timed solve_bracketed(P) with P already built, against
solve_spherical(...) which builds its own _Problem. _Problem tabulates the
baryon potential on a (n_steps+1) x n_gl grid, which is a fixed cost every
method pays, and on an unloaded machine it dominates all of them -- which is
why three methods doing 1x, 16x and 32x the solve work all reported 0.0174 s.

Reported here: total cost through solve_spherical (what a caller pays), and
solve-only cost with the problem pre-built (what the algorithms actually cost).
Single process, pinned, interleaved so drift hits every method equally.
"""
import os, statistics, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from jeans.fast.solver import _Problem, solve_spherical, GN
import jeans.fast.branch as B

os.sched_setaffinity(0, {os.sched_getaffinity(0).pop()})


def nfw(M200, c, r1, h=0.7, dc=200.0, Om=0.3, Ol=0.7):
    H0 = h * 100.0 * 1e-3
    rc = 3.0 * H0 ** 2 / (8.0 * np.pi * GN) * (Om + Ol)
    rs = rc * dc / 3.0 * c ** 3 / (np.log1p(c) - c / (1.0 + c))
    r_s = np.cbrt(3.0 * M200 / (dc * 4.0 * np.pi * rc)) / c
    x = r1 / r_s
    return rs / (x * (1 + x) ** 2), 4 * np.pi * rs * r_s ** 3 * (np.log1p(x) - x / (1 + x))


mn = lambda Md, a, b: (lambda r, t: -GN * Md / np.sqrt(
    r ** 2 * np.sin(t) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(t) ** 2)) ** 2))

rng = np.random.default_rng(31)
CASES = []
while len(CASES) < 40:
    M200 = 10 ** rng.uniform(11.3, 12.9); c = rng.uniform(6, 16)
    r1 = rng.uniform(4, 22); Md = 10 ** rng.uniform(9.5, 11.0)
    a = rng.uniform(1.5, 5.0); b = rng.uniform(0.15, 0.9)
    rho1, M1 = nfw(M200, c, r1)
    pb = mn(Md, a, b)
    if not all(solve_spherical(r1, rho1, M1, Phi_b=pb, method=m,
                               verify=(m != "ramp")).success
               for m in ("bracket", "ramp", "ramp+schedule")):
        continue
    CASES.append((r1, rho1, M1, pb))
print(f"{len(CASES)} cases solvable by every method\n", flush=True)

METHODS = [
    ("plain Newton",          dict(method="ramp", n_ramp=1, verify=False)),
    ("16-stage ramp",         dict(method="ramp", verify=False)),
    ("ramp + certificate",    dict(method="ramp", verify=True)),
    ("ramp + schedule screen", dict(method="ramp+schedule", verify=True)),
    ("bracketed (default)",   dict(method="bracket")),
]

for name, kw in METHODS:                       # warm the numba kernels
    for c_ in CASES[:3]:
        solve_spherical(c_[0], c_[1], c_[2], Phi_b=c_[3], **kw)

REPS = 9
full = {n: [] for n, _ in METHODS}
for _ in range(REPS):
    for name, kw in METHODS:
        t0 = time.perf_counter()
        for r1, rho1, M1, pb in CASES:
            solve_spherical(r1, rho1, M1, Phi_b=pb, **kw)
        full[name].append((time.perf_counter() - t0) / len(CASES) * 1e3)

# fixed tabulation cost, measured on its own
t0 = time.perf_counter()
for _ in range(REPS):
    for r1, rho1, M1, pb in CASES:
        _Problem(r1, rho1, M1, 200, pb, 16, True)
tab = (time.perf_counter() - t0) / (REPS * len(CASES)) * 1e3

print(f"{'method':<24}{'total ms':>11}{'IQR':>9}{'minus setup':>13}")
for name, _ in METHODS:
    s = sorted(full[name]); med = statistics.median(s)
    iqr = s[3 * len(s) // 4] - s[len(s) // 4]
    print(f"{name:<24}{med:11.2f}{iqr:9.2f}{med - tab:13.2f}")
print(f"\n_Problem construction (paid by every method): {tab:.2f} ms")
print(f"cores visible {os.cpu_count()}, affinity {len(os.sched_getaffinity(0))}, "
      f"load {os.getloadavg()[0]:.1f}")
