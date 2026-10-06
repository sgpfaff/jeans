"""jeanie in one file: solve, certify, differentiate, export.

Runnable, and run by the test suite (tests/test_examples.py), so it cannot
drift from the code the way a README snippet does. Execute it directly:

    PYTHONPATH=src python examples/jeanie_quickstart.py
"""
import numpy as np

GN = 4.302e-6  # km^2/s^2 kpc / Msun


def miyamoto_nagai(Md, a, b, xp=np):
    """A disc, as Phi_b(r, theta). Pass xp=jnp for the differentiable path."""
    return lambda r, th: -GN * Md / xp.sqrt(
        r ** 2 * xp.sin(th) ** 2
        + (a + xp.sqrt(b ** 2 + r ** 2 * xp.cos(th) ** 2)) ** 2)


def spherical():
    """Solve the isothermal core matched to a CDM halo at r1."""
    from jeanie.outer import boundary_data
    from jeanie.solver import solve_spherical
    from jeans.classes import CDM_profile

    disc = miyamoto_nagai(6e10, 3.0, 0.28)
    halo = CDM_profile(1e12, 10.0, q0=1.0, Phi_b=disc)
    rho1, M1, _ = boundary_data(halo, 12.0, L_list=(0,))

    res = solve_spherical(12.0, rho1, M1, Phi_b=disc)
    print(f"  r0 = {res.r0:.4f} kpc   sigma0 = {res.sigma0:.2f} km/s   "
          f"[{res.reason}]")
    return res


def existence_and_branch():
    """The matching problem is multi-valued; ask before you trust a root.

    Below r1/rs = 1.780 the root is unique. Between 1.780 and 3.974 a solution
    exists but there is more than one, so which branch you land on matters.
    Beyond 3.974 there is none at all.
    """
    from jeanie import branch, universal
    from jeanie.outer import boundary_data
    from jeanie.solver import _Problem, solve_spherical
    from jeans.classes import CDM_profile

    disc = miyamoto_nagai(6e10, 3.0, 0.28)
    for r1 in (6.0, 12.0, 30.0):
        halo = CDM_profile(1e12, 10.0, q0=1.0, Phi_b=disc)
        rho1, M1, _ = boundary_data(halo, r1, L_list=(0,))
        R = universal.matching_ratio_of(r1, rho1, M1)
        res = solve_spherical(r1, rho1, M1, Phi_b=disc)
        P = _Problem(r1, rho1, M1, 200, disc, 16, True)
        ok, why = (branch.certify(P, res.r0, res.sigma0) if res.success
                   else (False, res.reason))
        print(f"  r1 = {r1:5.1f}  R = {R:.4f}  (R_MAX = {universal.R_MAX:.4f})"
              f"  solved = {res.success}  certificate = {ok} [{why}]")


def axisymmetric():
    """A flattened outer halo drives L>0 structure in the core."""
    from jeanie.outer import boundary_data
    from jeanie.solver2d import solve_axisymmetric
    from jeans.classes import CDM_profile

    disc = miyamoto_nagai(6e10, 3.0, 0.28)
    halo = CDM_profile(1e12, 10.0, q0=0.8, Phi_b=disc)
    rho1, M1, J_L = boundary_data(halo, 10.0, L_list=(0, 2))

    res = solve_axisymmetric(10.0, rho1, M1, J_L=J_L, L_list=(0, 2), Phi_b=disc)
    print(f"  r0 = {res.r0:.4f} kpc   phi_2(0.99 r1) = {res.phi_at(2, 9.9):+.5f}")
    print(f"  max|psi| = {res.max_psi:.3f}  "
          f"estimated shape error ~ {res.shape_rel_error:.1e}")


def differentiable():
    """Gradients in the halo parameters, by implicit differentiation."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jeanie.jaxsolver import solve_log

    p = jnp.array([1e12, 10.0, 12.0, 6e10, 3.0, 0.28])  # M200 c r1 Md a b
    lg = solve_log(p)
    print(f"  r0 = {float(jnp.exp(lg[0])):.4f}   "
          f"sigma0 = {float(jnp.exp(lg[1])):.2f}")
    jac = jax.jacfwd(solve_log)(p)
    print(f"  d(log r0)/d(log c) = "
          f"{float(jac[0, 1] * p[1]):+.4f}   (2x6 Jacobian, one pass)")
    # a whole ensemble at once
    P = jnp.tile(p, (16, 1)).at[:, 0].set(jnp.linspace(5e11, 2e12, 16))
    print(f"  vmap over 16 halos -> r0 from "
          f"{float(jnp.exp(jax.vmap(solve_log)(P)[0, 0])):.3f} to "
          f"{float(jnp.exp(jax.vmap(solve_log)(P)[-1, 0])):.3f} kpc")


def contracted_boundary():
    """Einasto and adiabatic contraction, in closed form and differentiable."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jeanie import jaxouter as JO

    disc = miyamoto_nagai(5e10, 2.5, 0.4, xp=jnp)   # traced: must be jnp
    for label, kw in (("NFW", {}),
                      ("Einasto a=0.18", dict(halo_type="Einasto", alpha=0.18)),
                      ("NFW + Cautun AC",
                       dict(AC_prescription="Cautun", Phi_b=disc))):
        rho1, M1, _ = JO.boundary_data(1e12, 10.0, 10.0, q0=0.85, L_list=(0,),
                                       **kw)
        print(f"  {label:18s} rho1 = {float(rho1):.4e}   M1 = {float(M1):.4e}")


def export_as_a_potential():
    """What a stream integrator needs: M(<r) and the radial acceleration.

    No Poisson solve and no differentiating an interpolated potential -- the
    march's second variable gives M(<r) in closed form, and continuity at r1
    is a matching condition, so it holds to machine precision.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jeanie import jaxprofile as JP
    from jeanie.jaxsolver import nfw_boundary

    p = jnp.array([1e12, 10.0, 12.0, 6e10, 3.0, 0.28])
    M = JP.mass_fn(p)
    g = JP.radial_acceleration_fn(p)
    rho = JP.density_fn(p)

    M1 = float(nfw_boundary(1e12, 10.0, 12.0)[1])
    print(f"  M(<r1)/M1 - 1 = {abs(float(M(12.0)) / M1 - 1):.2e}  "
          f"(a matching condition, so exact)")
    for r in (2.0, 8.0, 20.0):
        print(f"    r = {r:5.1f}  M = {float(M(r)):.4e} Msun   "
              f"g = {float(g(r)):+.2f}   rho(R=r,z=0) = {float(rho(r, 0.0)):.4e}")
    # and it is differentiable, which is the point
    dg = jax.grad(lambda q: JP.radial_acceleration_fn(q)(8.0))(p)
    print(f"  d g(8 kpc) / d log c = {float(dg[1] * p[1]):+.4f}")


def main():
    for name, fn in (("spherical solve", spherical),
                     ("existence and branch", existence_and_branch),
                     ("axisymmetric", axisymmetric),
                     ("differentiable solve", differentiable),
                     ("contracted boundary", contracted_boundary),
                     ("export as a potential", export_as_a_potential)):
        print(f"\n=== {name} ===")
        fn()


if __name__ == "__main__":
    main()
