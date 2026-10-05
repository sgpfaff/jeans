from .classes import profile, CDM_profile, isothermal_profile
from . import tools
from . import spherical as sphmodel
from . import nonspherical
from .tools import timed

# Generate profile object from inputs


def _spherical_baryon_input(outer_halo, baryon_average):
    """Pick the baryon potential handed to the spherical relaxation.

    "exact"     -- pass Phi_b(r, theta) through, so the solver forms the
                   rigorous m0 = <exp(-Phi_b/sigma0^2)>. Matches jeans.isothermal.
    "spherical" -- pre-average the potential and pass Phi_b(r), giving
                   m0 = exp(-<Phi_b>/sigma0^2). The original behaviour.
    """
    if baryon_average == "exact":
        return outer_halo.Phi_b
    elif baryon_average == "spherical":
        return tools.compute_Phi_b_spherical(
            outer_halo.M_b, 1e-6 * outer_halo.r200, outer_halo.r200
        )
    raise Exception(
        "baryon_average=%r not found; use 'exact' or 'spherical'." % (baryon_average,)
    )


# Spherical isothermal Jeans model
@timed
def spherical(
    r1, *outer_halo_params, Phi_b=None, halo_type="NFW", gamma=0.3,
    baryon_average="exact", **kwargs
):

    # CDM profile
    if r1 == 0:
        outer_halo = CDM_profile(*outer_halo_params, q0=1, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)
        return profile(outer=outer_halo)

    # Spherical Jeans profile
    else:

        # CDM outer profile (spherical)
        outer_halo = CDM_profile(*outer_halo_params, q0=1, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)

        # D6: jeans.isothermal passes the full Phi_b, so its spherical step uses
        # the rigorous m0 = <exp(-Phi_b/sigma0^2)>, while this path used
        # m0 = exp(-<Phi_b>/sigma0^2). For an aspherical Phi_b the two differ by
        # a Jensen gap -- up to ~1.8% in r0 for a Miyamoto-Nagai disc -- so
        # jeans.spherical was not the L=[0] limit of jeans.isothermal.
        # Default to the rigorous convention; the old one stays reachable.
        Phi_b_in = _spherical_baryon_input(outer_halo, baryon_average)

        # Spherical Jeans model matched onto the spherical outer halo
        inner_halo, success = sphmodel.relaxation(r1, outer_halo, Phi_b=Phi_b_in, **kwargs)

        # Matching was successful
        if success:
            halo = profile(inner=inner_halo, outer=outer_halo)
            return halo

        # Unsuccessful matching
        else:
            return None


# Squashed Jeans model
@timed
def squashed(
    r1, *outer_halo_params, q0=1, Phi_b=None, halo_type="NFW", gamma=0.3,
    q_mode="smooth", baryon_average="exact", **kwargs
):

    # CDM profile
    if r1 == 0:
        return cdm(*outer_halo_params, q0=q0, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)
    # Squashed Jeans profile
    # Start with spherical profiles
    else:

        # CDM outer profile (spherical)
        outer_halo = CDM_profile(*outer_halo_params, q0=1, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)

        # D6: jeans.isothermal passes the full Phi_b, so its spherical step uses
        # the rigorous m0 = <exp(-Phi_b/sigma0^2)>, while this path used
        # m0 = exp(-<Phi_b>/sigma0^2). For an aspherical Phi_b the two differ by
        # a Jensen gap -- up to ~1.8% in r0 for a Miyamoto-Nagai disc -- so
        # jeans.spherical was not the L=[0] limit of jeans.isothermal.
        # Default to the rigorous convention; the old one stays reachable.
        Phi_b_in = _spherical_baryon_input(outer_halo, baryon_average)

        # Spherical Jeans model matched onto the spherical outer halo
        inner_halo, success = sphmodel.relaxation(r1, outer_halo, Phi_b=Phi_b_in, **kwargs)

        # Matching was successful
        if success:

            if q_mode == "uniform":

                # Apply uniform squashing by q0
                halo = profile(inner=inner_halo, outer=outer_halo, q=q0)
                return halo

            elif q_mode == "smooth":

                # Calculate q(r_sph) from spherical Jeans model profile
                sph_halo = profile(inner=inner_halo, outer=outer_halo)
                q_eff = tools.compute_q_eff(sph_halo, q0, **kwargs)

                halo = profile(inner=inner_halo, outer=outer_halo, q=q_eff)
                return halo

            elif q_mode == "old":

                # Calculate q(r_sph) from spherical Jeans model profile using old method (fit ansatz)
                sph_halo = profile(inner=inner_halo, outer=outer_halo)
                q_eff = tools.compute_q_eff(sph_halo, q0, **kwargs)

                halo = profile(inner=inner_halo, outer=outer_halo, q=q_eff)
                return halo

            else:
                raise Exception("Unknown q_mode=%s.")

        # Unsuccessful matching
        else:
            return None


# Nonspherical isothermal Jeans model
@timed
def isothermal(r1, *outer_halo_params, q0=1, Phi_b=None, halo_type="NFW", gamma=0.3, **kwargs):

    # CDM profile
    if r1 == 0:
        return cdm(*outer_halo_params, q0=q0, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)
    # Isothermal nonspherical Jeans profile
    else:

        # CDM outer halo
        outer_halo = CDM_profile(*outer_halo_params, q0=q0, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)

        # Nonspherical Jeans model matched onto nonspherical outer halo
        inner_halo, success = nonspherical.relaxation(r1, outer_halo, Phi_b=outer_halo.Phi_b, **kwargs)

        # Matching was successful
        if success:
            return profile(inner=inner_halo, outer=outer_halo)

        # Unsuccessful matching
        else:
            return None


# CDM profile
@timed
def cdm(*outer_halo_params, q0=1, Phi_b=None, halo_type="NFW", gamma=0.3, **kwargs):

    # Spherically symmetric CDM halo
    outer_halo = CDM_profile(*outer_halo_params, Phi_b=Phi_b, halo_type=halo_type, gamma=gamma, **kwargs)

    # Return profile object squashed by q0
    return profile(outer=outer_halo, q=q0)
