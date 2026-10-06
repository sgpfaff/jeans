"""What does adiabatic contraction do to the inferred SIDM core radius?

The density shift at r1 is a statement about the boundary condition. The
quantity a stream fit actually constrains is r0, so measure that: solve the
same halo with AC off and on, and compare.
"""
import os, sys, warnings
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
warnings.simplefilter("ignore")
from jeans.definitions import GN
from jeans.fast import jaxouter as JO
from jeans.fast.solver import solve_spherical

rng = np.random.default_rng(17)
rows = {"Cautun": [], "Gnedin": []}
n_tried = 0
while n_tried < 120:
    M200 = 10 ** rng.uniform(11.5, 12.8); c = rng.uniform(6, 14)
    Md = 10 ** rng.uniform(9.5, 11.0); a = rng.uniform(1.5, 5.0)
    b = rng.uniform(0.15, 0.9); r1 = rng.uniform(5, 30); q0 = rng.uniform(0.7, 1.0)
    mj = (lambda M, A, B: lambda r, th: -GN * M / jnp.sqrt(
        r ** 2 * jnp.sin(th) ** 2 + (A + jnp.sqrt(B ** 2 + r ** 2 * jnp.cos(th) ** 2)) ** 2))(Md, a, b)
    mn = (lambda M, A, B: lambda r, th: -GN * M / np.sqrt(
        r ** 2 * np.sin(th) ** 2 + (A + np.sqrt(B ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2))(Md, a, b)
    n_tried += 1
    try:
        rho0, M0, _ = JO.boundary_data(M200, c, r1, q0=q0, L_list=(0,))
        base = solve_spherical(r1, float(rho0), float(M0), Phi_b=mn)
        if not base.success:
            continue
        for presc in ("Cautun", "Gnedin"):
            rhoA, MA, _ = JO.boundary_data(M200, c, r1, q0=q0, L_list=(0,),
                                           AC_prescription=presc, Phi_b=mj)
            ac = solve_spherical(r1, float(rhoA), float(MA), Phi_b=mn)
            if ac.success:
                rows[presc].append((ac.r0 / base.r0 - 1.0,
                                    ac.sigma0 / base.sigma0 - 1.0,
                                    ac.rho0 / base.rho0 - 1.0))
    except Exception:
        continue

print(f"{n_tried} halos drawn; AC on vs AC off, both solved\n")
for presc in ("Cautun", "Gnedin"):
    A = np.array(rows[presc]) * 100
    print(f"{presc}  (n = {len(A)})")
    for k, nm in enumerate(("core radius r0", "dispersion sigma0", "central density rho0")):
        print(f"   {nm:22s} median {np.median(A[:,k]):+7.1f}%   "
              f"p10 {np.percentile(A[:,k],10):+7.1f}%   p90 {np.percentile(A[:,k],90):+7.1f}%"
              f"   |max| {np.max(np.abs(A[:,k])):6.1f}%")
