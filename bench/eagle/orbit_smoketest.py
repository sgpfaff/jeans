"""Does a stream integrator's vector field actually work off jeanie?

The seam a stream code needs is an acceleration, called millions of times per
likelihood. jeanie gives it in closed form -- the march's second variable is
the enclosed mass -- so there is no Poisson solve and nothing to differentiate
numerically. This integrates a real orbit in that field with diffrax (which is
what both StreamSculptor and galax are built on) and differentiates the result
end to end, through the orbit AND through the implicit halo solve.

Note the field here is the DARK MATTER only. A real stream needs the total,
so the baryon acceleration must be added; see jaxprofile.radial_acceleration_fn.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import diffrax as dfx
from jeanie import jaxprofile as JP

P = jnp.array([1e12, 10.0, 12.0, 6e10, 3.0, 0.28])   # M200 c r1 Md a b


def make_field(params):
    g = JP.radial_acceleration_fn(params)

    def field(t, y, args):
        x, v = y[:3], y[3:]
        r = jnp.sqrt(jnp.sum(x ** 2) + 1e-12)
        return jnp.concatenate([v, g(r) * x / r])

    return field


def integrate(params, y0, t1=2.05):
    """Units: kpc and km/s, so with G = 4.302e-6 km^2/s^2 kpc/Msun the
    acceleration is in (km/s)^2/kpc and time is in kpc/(km/s) = 0.978 Gyr.
    t1 = 2.05 is therefore 2 Gyr. Getting this wrong asks the integrator for
    2000 Gyr at 1/1000th the orbital speed, which it answers with max_steps.
    """
    term = dfx.ODETerm(make_field(params))
    sol = dfx.diffeqsolve(
        term, dfx.Dopri5(), t0=0.0, t1=t1, dt0=0.5, y0=y0,
        stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-9),
        max_steps=200_000,
        saveat=dfx.SaveAt(ts=jnp.linspace(0.0, t1, 64)),
        adjoint=dfx.DirectAdjoint())      # the only one supporting BOTH modes
    return sol


if __name__ == "__main__":
    # An orbit that goes INSIDE r1 = 12 kpc, so the inner solve is actually
    # exercised. An orbit that stays outside it has identically zero gradient
    # with respect to r1, Md, a and b, because out there the field is just the
    # outer halo -- a useful check, but not a test of the seam.
    y0 = jnp.array([14.0, 0.0, 2.0, 0.0, 110.0, 20.0])   # kpc, km/s
    t0 = time.time(); sol = integrate(P, y0)
    r = np.sqrt(np.sum(np.asarray(sol.ys)[:, :3] ** 2, axis=1))
    print(f"orbit integrated in {time.time()-t0:.2f} s (incl. compile)")
    print(f"  r from {r.min():.2f} to {r.max():.2f} kpc over 2 Gyr, "
          f"{np.isfinite(r).sum()}/{len(r)} finite")

    # energy conservation is the honest check that the field is right
    def loss(params):
        s = integrate(params, y0)
        return jnp.sqrt(jnp.sum(s.ys[-1, :3] ** 2))      # final radius

    t0 = time.time(); val = float(loss(P)); print(f"  final radius {val:.6f} kpc")
    t0 = time.time(); gf = jax.jacfwd(loss)(P)
    print(f"  jacfwd through orbit + custom_root: {time.time()-t0:.1f} s")
    print("   ", np.array(gf))
    t0 = time.time(); gr = jax.grad(loss)(P)
    print(f"  jax.grad (reverse):                 {time.time()-t0:.1f} s")
    rel = np.max(np.abs(np.asarray(gr) - np.asarray(gf))
                 / (np.abs(np.asarray(gf)) + 1e-30))
    print(f"  forward vs reverse agree to {rel:.2e}  "
          f"(they share no code in diffrax)")
