"""Outer-halo boundary data in closed form, differentiable in the halo parameters.

:mod:`jeans.fast.outer` made the boundary fast by replacing nested adaptive
quadrature with fixed nodes, but it still calls the package's ``rho_sph`` on
those nodes, which is a host function. That blocks ``vmap`` and, more
importantly, blocks gradients: a stream fit varies M200, c and q0, and those
are exactly the parameters the spline path cannot be differentiated through.

Adiabatic contraction is not an optional refinement for that fit. Measured
over a realistic halo+disc prior at r1 = 5-30 kpc, switching it on moves the
dark-matter density by a median -11.5% (Cautun) or -14.0% (Gnedin), with a
p10-p90 spread of -18.5% to +14%, and the sign depends on the configuration.
Part of that is bookkeeping rather than contraction: the Cautun factor at zero
baryon mass is 0.45 + 0.38 * 1.16**0.53 = 0.861, not 1, because M_CDM there is
the total-matter profile and the prescription removes the cosmological baryon
fraction. So AC-on and AC-off are different models of the same halo, not a
small correction to one.

What that does to the solved core, over 120 halos from the same prior: the
core radius r0 moves by a median -13.9% (Cautun) or -10.7% (Gnedin), with a
p10 of -36% and -40% and a maximum of 56% and 60%, and the central density by
a median +24.9% and +7.7% with a p90 of +148% and +172%. r0 is essentially
what constrains the cross-section, so running without contraction is a
modelling choice to be stated rather than a simplification.

Here the density itself is analytic, so the same fixed-node quadratures become
one traceable expression and the derivatives come out exactly.

The squashing is what makes this possible at all. The package defines the
squashed radius by a fixed point,

    r_sph = sqrt(R^2 q(r_sph)^(2/3) + z^2 q(r_sph)^(-4/3))

and iterates it. For a **constant** q the iteration is a no-op and the map is
closed form:

    r_sph(r, theta) = r sqrt(sin^2(theta) q^(2/3) + cos^2(theta) q^(-4/3))

which is differentiable in q0 directly. A radius-dependent q(r) would need the
fixed point unrolled to a fixed depth; not implemented, and the functions here
raise rather than pretend.

Scope: NFW or Einasto, constant q0, optionally adiabatically contracted by
either the Cautun or the Gnedin prescription. Everything is analytic or an
implicitly differentiated root-find, so gradients reach M200, c, q0 and the
Einasto shape index alpha.

Three things the package does numerically are done exactly here, and all three
were costing smoothness rather than only speed:

  * The baryon enclosed mass. The package builds M_b(r) = r^2/G dPhi_b/dr by
    splining the spherically averaged potential on 100 log-spaced points and
    differentiating the spline. Here the angular average is a fixed
    Gauss-Legendre rule and the radial derivative is autodiff, so M_b is exact.

  * The contracted density. The package recovers rho from M by np.gradient of
    log M on a 1000-point grid and splines the result. Here rho = M'(r)/4 pi r^2
    is autodiff of the closed-form mass. This also removes the failure behind
    the second half of defect D2, where Einasto's finite total mass made the
    numerical dlogM/dlogr collapse to round-off in the saturated tail and
    return NaN at every radius -- an analytic derivative has no such tail
    problem.

  * The Gnedin initial radius. The package iterates find_ri to a relative
    tolerance with a damping factor and raises after 1000 iterations, which is
    neither traceable nor differentiable. Here the same fixed point is wrapped
    in lax.custom_root, so the forward pass runs a fixed number of damped
    steps and the derivative comes from the implicit function theorem exactly,
    independent of how many steps were taken.

Cautun needs none of that: despite the package tabulating it on 1000 points,
the prescription is closed form in M_CDM(r) and M_b(r).
"""
import numpy as np

from .jaxsolver import GN, HAVE_JAX, N_GL_DEFAULT, _require_jax, gl_tables

if HAVE_JAX:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammainc, gammaln

__all__ = ["nfw_params", "nfw_rho", "nfw_mass", "einasto_params", "einasto_rho",
           "einasto_mass", "mb_spherical", "ac_mass", "halo_profile", "r_sph",
           "rho_sph_avg", "enclosed_mass", "potential_moments", "boundary_data",
           "F_B"]

# Cosmological baryon fraction, matching the value hardcoded in
# jeans.cdm.AC_profiles (which notes it is Cautun et al.'s, not an estimate
# from the supplied baryon profile).
F_B = 0.156352

N_R_DEFAULT = 64


def nfw_params(M200, c, h=0.7, del_c=200.0, Om=0.3, Ol=0.7, z=0.0):
    """(rho_s, r_s), matching jeans.cdm.mass_concentration_to_NFW_parameters."""
    H0 = h * 100.0 * 1e-3
    rho_crit = (3.0 * H0 ** 2) / (8.0 * jnp.pi * GN) * (Om * (1.0 + z) ** 3 + Ol)
    over = (del_c / 3.0) * (c ** 3) / (jnp.log1p(c) - c / (1.0 + c))
    rvir = jnp.cbrt(3.0 * M200 / (del_c * 4.0 * jnp.pi * rho_crit))
    return rho_crit * over, rvir / c


def nfw_rho(r, rho_s, r_s):
    x = r / r_s
    return rho_s / (x * (1.0 + x) ** 2)


def nfw_mass(r, rho_s, r_s):
    x = r / r_s
    return 4.0 * jnp.pi * rho_s * r_s ** 3 * (jnp.log1p(x) - x / (1.0 + x))


# --------------------------------------------------------------------------
# Einasto
# --------------------------------------------------------------------------
def _m_c_alpha(c, alpha):
    """The package's mass factor m(c, alpha), jeans.cdm._m_c_alpha."""
    a = 3.0 / alpha
    x = (2.0 / alpha) * c ** alpha
    lower_inc = jnp.exp(gammaln(a)) * gammainc(a, x)
    prefac = jnp.exp(2.0 / alpha) / alpha * (2.0 / alpha) ** (-3.0 / alpha)
    return prefac * lower_inc


def einasto_params(M200, c, alpha, h=0.7, del_c=200.0, Om=0.3, Ol=0.7, z=0.0):
    """(rho_minus2, r_minus2, r200), matching mass_concentration_to_Einasto_parameters."""
    H0 = h * 100.0 * 1e-3
    rho_crit = (3.0 * H0 ** 2) / (8.0 * jnp.pi * GN) * (Om * (1.0 + z) ** 3 + Ol)
    rvir = jnp.cbrt(3.0 * M200 / (4.0 * jnp.pi * del_c * rho_crit))
    r_m2 = rvir / c
    return M200 / (4.0 * jnp.pi * r_m2 ** 3 * _m_c_alpha(c, alpha)), r_m2, rvir


def einasto_rho(r, rho_m2, r_m2, alpha):
    x = r / r_m2
    return rho_m2 * jnp.exp(-(2.0 / alpha) * (x ** alpha - 1.0))


def einasto_mass(r, M200, c, alpha, **kw):
    """M(<r) for an Einasto halo. Normalised so that M(r200) = M200.

    jax.scipy.special.gammainc is differentiable in the ORDER as well as the
    argument -- checked against finite differences at (3.0, 2.5), -0.24711735748
    against -0.24711735766 -- so alpha is an inferable parameter here and not
    merely a setting.
    """
    _, r_m2, _ = einasto_params(M200, c, alpha, **kw)
    sa = 3.0 / alpha
    x = (2.0 / alpha) * (r / r_m2) ** alpha
    return M200 * gammainc(sa, x) / gammainc(sa, (2.0 / alpha) * c ** alpha)


# --------------------------------------------------------------------------
# baryons and adiabatic contraction
# --------------------------------------------------------------------------
def _phi_b_sph_avg(Phi_b, r, n_gl):
    """< Phi_b >_solid angle. The package's 0.5 * int_0^pi Phi_b sin(th) dth.

    Phi_b must be written with jax.numpy, not numpy: it is differentiated and
    so is called on tracers. The arity is read from the signature rather than
    guessed by catching TypeError, which silently turned a tracer error into a
    wrong-number-of-arguments error one frame later.
    """
    from inspect import signature
    n_args = len(signature(Phi_b).parameters)
    if n_args == 1:
        return Phi_b(r)
    if n_args != 2:
        raise ValueError("Phi_b must take (r) or (r, theta); got %d arguments"
                         % n_args)
    theta, w = _angular(n_gl)
    return jnp.sum(w * Phi_b(r, theta))


def mb_spherical(Phi_b, r, n_gl=N_GL_DEFAULT):
    """M_b(r) = r^2/G d<Phi_b>/dr, by autodiff rather than by splining.

    The package splines the averaged potential on 100 log-spaced points and
    differentiates the spline; this is the same quantity computed exactly.
    """
    d = jax.grad(lambda rr: _phi_b_sph_avg(Phi_b, rr, n_gl))
    return r ** 2 / GN * d(r)


def _bar(r, r200, A0, w):
    """Gnedin's orbit-averaged radius, bar(r)/r0 = A0 (r/r0)^w with r0 = 0.03 r200."""
    r0 = 0.03 * r200
    return A0 * r0 * (r / r0) ** w


def ac_mass(r, M_cdm, M_b, prescription="Cautun", r200=None,
            Gnedin_params=(1.6, 0.8), n_newton=40):
    """Adiabatically contracted dark-matter mass inside r.

    `M_cdm` and `M_b` are callables of radius. Cautun is closed form. Gnedin
    solves its fixed point for the initial radius r_i under lax.custom_root,
    so the forward pass is a fixed number of damped steps and the derivative
    is exact by the implicit function theorem regardless.
    """
    if prescription == "Cautun":
        mc = M_cdm(r)
        return mc * (0.45 + 0.38 * ((1.0 - F_B) / F_B * M_b(r) / mc + 1.16) ** 0.53)
    if prescription != "Gnedin":
        raise ValueError("prescription must be 'Cautun' or 'Gnedin'; got %r"
                         % (prescription,))
    A0, w = Gnedin_params

    def resid(ri):
        return ri - r * (1.0 - F_B) * (
            1.0 + M_b(_bar(r, r200, A0, w)) / M_cdm(_bar(ri, r200, A0, w)))

    def solve(f, x0):
        def step(x, _):
            # the package's damped update, weight 0.5, at a fixed step count
            return 0.5 * (x - f(x)) + 0.5 * x, None
        x, _ = jax.lax.scan(step, x0, None, length=n_newton)
        return x

    def tangent_solve(g, y):
        return y / jax.grad(lambda t: g(t))(1.0)

    ri = jax.lax.custom_root(resid, r, solve, tangent_solve)
    return M_cdm(ri)


# --------------------------------------------------------------------------
# dispatch, matching jeans.classes.CDM_profile
# --------------------------------------------------------------------------
def halo_profile(M200, c, halo_type="NFW", alpha=0.18, AC_prescription=None,
                 Phi_b=None, Gnedin_params=(1.6, 0.8), n_gl=N_GL_DEFAULT):
    """Return rho_sph(r), the spherical dark-matter density, as one traced expression.

    Mirrors CDM_profile's dispatch on (halo_type, AC_prescription). With
    contraction the density is recovered as M'(r)/(4 pi r^2) by autodiff of the
    closed-form mass, not by differencing a spline.
    """
    _require_jax()
    if halo_type == "NFW":
        rho_s, r_s = nfw_params(M200, c)
        _, _, r200 = einasto_params(M200, c, alpha)   # same overdensity relation
        M_cdm = lambda r: nfw_mass(r, rho_s, r_s)
        rho_un = lambda r: nfw_rho(r, rho_s, r_s)
    elif halo_type == "Einasto":
        rho_m2, r_m2, r200 = einasto_params(M200, c, alpha)
        M_cdm = lambda r: einasto_mass(r, M200, c, alpha)
        rho_un = lambda r: einasto_rho(r, rho_m2, r_m2, alpha)
    else:
        raise ValueError("halo_type must be 'NFW' or 'Einasto'; got %r" % (halo_type,))

    if AC_prescription is None:
        return rho_un
    if Phi_b is None:
        raise ValueError("adiabatic contraction needs Phi_b to build M_b(r)")
    M_b = lambda r: mb_spherical(Phi_b, r, n_gl)
    M_ac = lambda r: ac_mass(r, M_cdm, M_b, AC_prescription, r200, Gnedin_params)
    dM = jax.grad(M_ac)
    return lambda r: dM(r) / (4.0 * jnp.pi * r ** 2)


def r_sph(r, theta, q0):
    """The package's squashed radius, closed form for constant q.

    r_sph = r sqrt(sin^2(theta) q^(2/3) + cos^2(theta) q^(-4/3))
    """
    st2 = jnp.sin(theta) ** 2
    ct2 = jnp.cos(theta) ** 2
    return r * jnp.sqrt(st2 * q0 ** (2.0 / 3.0) + ct2 * q0 ** (-4.0 / 3.0))


def _angular(n_gl):
    theta, w = gl_tables(n_gl)
    return jnp.asarray(theta), jnp.asarray(w)


def rho_sph_avg(r, M200, c, q0=1.0, n_gl=N_GL_DEFAULT, rho_sph=None, **kw):
    """< rho(r, theta) >_theta for a squashed halo.

    `rho_sph` is the spherical density as a function of radius; omitted, it is
    built from (M200, c) and any halo_profile keywords given in **kw.
    """
    f = halo_profile(M200, c, **kw) if rho_sph is None else rho_sph
    theta, w = _angular(n_gl)
    return jnp.sum(w * _ev(f, r_sph(r, theta, q0)))


def enclosed_mass(r1, M200, c, q0=1.0, n_r=48, n_gl=16, rho_sph=None, **kw):
    """M(<r1) for a squashed NFW halo.

    Uses r = r1 v^2, which turns the r^-1 central cusp of
    4 pi r^2 rho dr into a smooth 8 pi r1^3 v^5 rho(r1 v^2) dv, so a fixed rule
    needs no inner cutoff and no separate tail integral. Same substitution as
    the numpy version in jeans.fast.outer, which agrees with the package to
    5.7e-11.
    """
    f = halo_profile(M200, c, **kw) if rho_sph is None else rho_sph
    xv, wv_np = np.polynomial.legendre.leggauss(n_r)
    v = jnp.asarray(0.5 * (xv + 1.0))
    wv = jnp.asarray(0.5 * wv_np)
    theta, wx = _angular(n_gl)

    rk = r1 * v * v                                        # (n_r,)
    rr = r_sph(rk[:, None], theta[None, :], q0)            # (n_r, n_gl)
    rho_avg = jnp.sum(wx[None, :] * _ev(f, rr), axis=1)
    return 8.0 * jnp.pi * r1 ** 3 * jnp.sum(wv * v ** 5 * rho_avg)


def potential_moments(r1, M200, c, q0=1.0, L_list=(0,), n_r=N_R_DEFAULT,
                      n_gl=24, rho_sph=None, **kw):
    """J_L for each L, matching CDM_profile.compute_potential_moments.

    Uses t = r1/r, which maps the infinite tail onto t -> 0 where the integrand
    tends to t^L / r1^(L+1) for an r^-3 profile, so one fixed rule covers the
    whole range with no shell sequence.
    """
    from scipy.special import eval_legendre
    L_list = tuple(L_list)
    if any(L % 2 for L in L_list):
        raise ValueError("odd L are not supported (z-symmetry assumed)")
    f = halo_profile(M200, c, **kw) if rho_sph is None else rho_sph

    xt, wt = np.polynomial.legendre.leggauss(n_r)
    t = jnp.asarray(0.5 * (xt + 1.0))
    wt = jnp.asarray(0.5 * wt)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    xg, wg = 0.5 * (xg + 1.0), 0.5 * wg
    theta = jnp.asarray(np.arccos(np.clip(xg, -1.0, 1.0)))
    wx = jnp.asarray(wg)
    Z = jnp.asarray([np.sqrt((2 * L + 1) / (4.0 * np.pi)) * eval_legendre(L, xg)
                     for L in L_list])

    r = r1 / t
    rr = r_sph(r[:, None], theta[None, :], q0)
    rho = _ev(f, rr)                                        # (n_r, n_gl)

    out = []
    for k, L in enumerate(L_list):
        rho_L = 4.0 * jnp.pi * jnp.sum(wx[None, :] * Z[k][None, :] * rho, axis=1)
        out.append(r1 ** (2 - L) * jnp.sum(wt * t ** (L - 3) * rho_L))
    return jnp.stack(out)


def _ev(f, x):
    """Evaluate a scalar-in/scalar-out density on an array of radii.

    halo_profile's contracted branch closes over jax.grad, which is scalar, so
    it cannot be broadcast directly; the uncontracted branches can. vmap twice
    for the (n_r, n_gl) grids either way -- it costs nothing when f is already
    elementwise and is required when it is not.
    """
    return jax.vmap(jax.vmap(f))(x) if jnp.ndim(x) == 2 else jax.vmap(f)(x)


def boundary_data(M200, c, r1, q0=1.0, L_list=(0,), n_r=N_R_DEFAULT,
                  n_gl=N_GL_DEFAULT, **kw):
    """(rho1, M1, J_L), all differentiable in M200, c, q0 and r1.

    For q0 == 1 this reproduces jaxsolver.nfw_boundary; for q0 != 1 it is the
    squashed halo the package builds, and it is the piece that was previously
    forcing a host call and cutting the gradient chain.
    """
    _require_jax()
    rho_sph = kw.pop("rho_sph", None)
    if rho_sph is None:
        rho_sph = halo_profile(M200, c, **kw)
    return (rho_sph_avg(r1, M200, c, q0, n_gl, rho_sph=rho_sph),
            enclosed_mass(r1, M200, c, q0, rho_sph=rho_sph),
            potential_moments(r1, M200, c, q0, L_list, n_r, rho_sph=rho_sph))
