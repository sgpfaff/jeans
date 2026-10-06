"""Controlled head-to-head of the three JAX prototypes.

The three agents each measured their own implementation, on their own parameter
sets, at different times under different machine load, and -- the real confound
-- at different configurations. A ships n_newton=4 and UNROLL=4; B and C ship
n_newton=3 and unroll=1. Comparing those headline numbers conflates
architecture with configuration.

This runs all three at matched settings (n_steps=200, n_gl=16, n_ramp=16,
n_newton=3), on identical parameter sets, interleaved in one process so load
drift hits every implementation equally, pinned to one core by the caller.
"""
import importlib.util
import os
import statistics
import sys
import time

import numpy as np

SCRATCH = "/tmp/claude-1003/-geir-data-scr-gabrielspace-jeans/c2b40140-c765-436f-8e08-af5382a50138/scratchpad"
sys.path.insert(0, "/geir_data/scr/gabrielspace/jeans-fast/src")


def load(name, path):
    """Import a module by explicit path; all three are called jaxjeans.py."""
    d = os.path.dirname(path)
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    sys.path.remove(d)
    return m


A = load("jj_A", f"{SCRATCH}/jax-A-purelax/jaxjeans.py")
B = load("jj_B", f"{SCRATCH}/jax-B-optimistix/jaxjeans.py")
C = load("jj_C", f"{SCRATCH}/jax-C-port/jaxjeans.py")

import jax
import jax.numpy as jnp

from jeans.definitions import GN
from jeanie.solver import solve_spherical

# Match configurations. A's module-level UNROLL is read at trace time.
A.UNROLL = 1

NEWTON, RAMP, STEPS, NGL = 3, 16, 200, 16

# Parameter sets: 12 well inside the safe region, so every implementation
# should solve all of them and the comparison is about cost, not robustness.
rng = np.random.default_rng(20)
CASES = []
while len(CASES) < 12:
    p = np.array([10 ** rng.uniform(11.3, 12.9), rng.uniform(6, 16),
                  rng.uniform(4, 22), 10 ** rng.uniform(9.5, 11.0),
                  rng.uniform(1.5, 5.0), rng.uniform(0.15, 0.9)])
    M200, c, r1, Md, a, b = p
    rho1, M1 = [float(v) for v in A.nfw_boundary(M200, c, r1)]
    ratio = M1 / (4 * np.pi * r1 ** 3 * rho1)
    if ratio > 0.7:
        continue
    ref = solve_spherical(r1, rho1, M1,
                          Phi_b=(lambda Md=Md, a=a, b=b: (lambda r, th:
                                 -GN * Md / np.sqrt(r ** 2 * np.sin(th) ** 2 +
                                 (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)))(),
                          verify=False)
    if not ref.success:
        continue
    CASES.append((p, rho1, M1, ref.r0, ref.sigma0))
print(f"{len(CASES)} matched parameter sets, all solvable by numba\n", flush=True)


def numba_fn(p, rho1, M1):
    M200, c, r1, Md, a, b = p
    pb = lambda r, th: -GN * Md / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (a + np.sqrt(b ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2)
    r = solve_spherical(r1, rho1, M1, Phi_b=pb, n_steps=STEPS, n_gl=NGL,
                        n_ramp=RAMP, verify=False)
    return r.r0, r.sigma0


fA = jax.jit(lambda p: A.solve_x(p, n_steps=STEPS, n_gl=NGL, n_ramp=RAMP,
                                 n_newton=NEWTON))
fB = jax.jit(lambda p: B.solve_log(p, n_steps=STEPS, n_gl=NGL, n_ramp=RAMP,
                                   n_newton=NEWTON))
_C = C.JaxSpherical(n_newt=NEWTON, n_ls=5)
fC = jax.jit(_C.forward_masked)

IMPLS = [
    ("A pure-lax",   lambda p, rho1, M1: np.exp(0.5 * np.asarray(fA(jnp.asarray(p))))),
    ("B custom_root", lambda p, rho1, M1: np.exp(np.asarray(fB(jnp.asarray(p)))[:2])),
    ("C port",       lambda p, rho1, M1: np.exp(np.asarray(fC(*[jnp.asarray(v) for v in p])[:2]))),
    ("numba",        numba_fn),
]

print("warming up (compiles)...", flush=True)
for name, f in IMPLS:
    t = time.perf_counter()
    for p, rho1, M1, _, _ in CASES[:2]:
        jax.block_until_ready(f(p, rho1, M1))
    print(f"  {name:14s} warm+compile {time.perf_counter()-t:6.2f} s", flush=True)

# ---- correctness at matched settings, against numba ----
print("\n=== agreement with numba, matched settings ===", flush=True)
for name, f in IMPLS[:3]:
    worst = 0.0
    for p, rho1, M1, r0_ref, s0_ref in CASES:
        out = f(p, rho1, M1)
        worst = max(worst, abs(float(out[0]) / r0_ref - 1.0))
    print(f"  {name:14s} worst |dr0/r0| = {worst:.2e}", flush=True)

# ---- interleaved timing ----
REPS = 15
samples = {name: [] for name, _ in IMPLS}
print(f"\n=== interleaved timing, {REPS} reps x {len(CASES)} cases ===", flush=True)
for rep in range(REPS):
    for name, f in IMPLS:
        t0 = time.perf_counter()
        for p, rho1, M1, _, _ in CASES:
            jax.block_until_ready(f(p, rho1, M1))
        samples[name].append((time.perf_counter() - t0) / len(CASES) * 1e3)

base = statistics.median(samples["numba"])
print(f"{'impl':<16}{'median':>10}{'min':>10}{'p25':>10}{'p75':>10}{'vs numba':>12}")
for name, _ in IMPLS:
    s = sorted(samples[name])
    med = statistics.median(s)
    q1, q3 = s[len(s) // 4], s[3 * len(s) // 4]
    print(f"{name:<16}{med:9.2f}m{s[0]:9.2f}m{q1:9.2f}m{q3:9.2f}m{base/med:11.2f}x")

print(f"\ncores visible: {os.cpu_count()}, affinity: {len(os.sched_getaffinity(0))}, "
      f"load {os.getloadavg()[0]:.2f}")


# ---- gradient cost, the criterion that actually decides this ----
print("\n=== gradient cost, matched settings ===", flush=True)
GRAD = [
    ("A pure-lax",   jax.jit(jax.jacrev(lambda p: A.solve_x(
        p, n_steps=STEPS, n_gl=NGL, n_ramp=RAMP, n_newton=NEWTON)))),
    ("B custom_root", jax.jit(jax.jacrev(lambda p: B.solve_log(
        p, n_steps=STEPS, n_gl=NGL, n_ramp=RAMP, n_newton=NEWTON)[:2]))),
    ("C port",       jax.jit(jax.jacrev(lambda p: jnp.stack(
        _C.forward(*[p[i] for i in range(6)])[:2])))),
]
FWD = dict(IMPLS[:3])
for name, g in GRAD:
    p0 = jnp.asarray(CASES[0][0])
    t = time.perf_counter()
    jax.block_until_ready(g(p0))
    tc = time.perf_counter() - t
    gs, fs = [], []
    for rep in range(9):
        t0 = time.perf_counter()
        for p, _, _, _, _ in CASES:
            jax.block_until_ready(g(jnp.asarray(p)))
        gs.append((time.perf_counter() - t0) / len(CASES) * 1e3)
        t0 = time.perf_counter()
        for p, rho1, M1, _, _ in CASES:
            jax.block_until_ready(FWD[name](p, rho1, M1))
        fs.append((time.perf_counter() - t0) / len(CASES) * 1e3)
    gm, fm = statistics.median(gs), statistics.median(fs)
    print(f"  {name:14s} fwd {fm:6.2f} ms   full 2x6 Jacobian {gm:6.2f} ms   "
          f"= {gm/fm:.2f}x fwd   (compile {tc:.1f} s)", flush=True)

# ---- vmap throughput, pinned ----
print("\n=== vmap(64) per halo, pinned ===", flush=True)
rngb = np.random.default_rng(1)
Pb = np.stack([10 ** rngb.uniform(11.3, 12.9, 64), rngb.uniform(6, 16, 64),
               rngb.uniform(4, 22, 64), 10 ** rngb.uniform(9.5, 11.0, 64),
               rngb.uniform(1.5, 5.0, 64), rngb.uniform(0.15, 0.9, 64)])
VM = [
    ("A pure-lax",   jax.jit(jax.vmap(lambda p: A.solve_x(
        p, n_steps=STEPS, n_gl=NGL, n_ramp=RAMP, n_newton=NEWTON))), Pb.T),
    ("B custom_root", jax.jit(jax.vmap(lambda p: B.solve_log(
        p, n_steps=STEPS, n_gl=NGL, n_ramp=RAMP, n_newton=NEWTON))), Pb.T),
    ("C port",       jax.jit(jax.vmap(_C.forward_masked)), Pb),
]
for name, vf, arg in VM:
    a = jnp.asarray(arg)
    t = time.perf_counter()
    jax.block_until_ready(vf(a) if name != "C port" else vf(*a))
    tc = time.perf_counter() - t
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        jax.block_until_ready(vf(a) if name != "C port" else vf(*a))
        ts.append((time.perf_counter() - t0) / 64 * 1e3)
    print(f"  {name:14s} {statistics.median(ts):6.2f} ms/halo  "
          f"(min {min(ts):.2f})   compile {tc:.1f} s", flush=True)
