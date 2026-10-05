"""Regression tests for defects found in the reference implementation.

Each test is written to FAIL on the unmodified package at commit 6c31ba3 and to
pass after the corresponding fix. The defect IDs match Table 3 of the scoping
write-up.

Run the whole file against the original to confirm it reproduces every defect:
    PYTHONPATH=/geir_data/scr/gabrielspace/jeans/src pytest tests/test_defects.py
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

import jeans
from jeans.classes import CDM_profile
from jeans.definitions import GN

# A compact but physically sensible early-type host: these are the cheapest
# parameters that still exercise the L>0 machinery.
M200, CONC, Q0, R1 = 1e13, 7.0, 0.8, 30.0
MSTAR, A_H = 5e10, 4.0


def hernquist_phi(r):
    """Hernquist potential, a one-variable Phi_b (the case that trips defect D1)."""
    return -GN * MSTAR / (r + A_H)


MD, A_D, B_D = 6e10, 3.0, 0.28


def mn_phi(r, th):
    """Miyamoto-Nagai disc: a two-variable Phi_b, needed for defect D6.

    For a spherically symmetric Phi_b the two averages <exp(-Phi_b)> and
    exp(-<Phi_b>) coincide identically, so D6 cannot be seen with hernquist_phi.
    """
    return -GN * MD / np.sqrt(
        r ** 2 * np.sin(th) ** 2
        + (A_D + np.sqrt(B_D ** 2 + r ** 2 * np.cos(th) ** 2)) ** 2
    )


# --------------------------------------------------------------------- D1
def test_d1_sph_sym_flag_refreshed_by_update():
    """classes.py:882 + :1241 -- the flag is set in __init__ and never recomputed.

    nonspherical.relaxation builds the profile with L_list=[0] and then calls
    update(L_list=...). The flag stays True, so every output accessor reports a
    spherically symmetric halo even though phi_2 was solved for.
    """
    h = jeans.isothermal(R1, M200, CONC, q0=Q0, Phi_b=hernquist_phi,
                         L_list=[0, 2], r_grid=60)
    assert h is not None, "fixture failed to converge; adjust parameters"

    assert h.inner.L_list == [0, 2]
    assert h.inner.sph_sym_flag is False or h.inner.sph_sym_flag == False, (
        "sph_sym_flag is still True after update(L_list=[0,2]); the solved "
        "quadrupole is being discarded by every output accessor"
    )


def test_d1_quadrupole_reaches_the_density():
    """The observable consequence of D1: rho(pole) == rho(equator) exactly."""
    h = jeans.isothermal(R1, M200, CONC, q0=Q0, Phi_b=hernquist_phi,
                         L_list=[0, 2], r_grid=60)
    assert h is not None

    r = 0.9 * R1
    ratio = h.rho_sph(r, 0.0) / h.rho_sph(r, np.pi / 2)
    assert abs(ratio - 1.0) > 1e-6, (
        f"rho(pole)/rho(equator) = {ratio!r} is exactly 1: the quadrupole was "
        "silently zeroed"
    )


def test_d1_rho_LM_quadrupole_is_not_noise():
    """rho_LM(2,0) returns ~1e-7 (round-off) instead of its true O(1e5) value."""
    h = jeans.isothermal(R1, M200, CONC, q0=Q0, Phi_b=hernquist_phi,
                         L_list=[0, 2], r_grid=60)
    assert h is not None

    r = 0.9 * R1
    rho0 = abs(h.inner.rho_LM(0, 0, r))
    rho2 = abs(h.inner.rho_LM(2, 0, r))
    assert rho2 / rho0 > 1e-4, (
        f"|rho_LM(2,0)/rho_LM(0,0)| = {rho2 / rho0:.3e} is round-off"
    )


# --------------------------------------------------------------------- D2
def test_d2_halo_type_respected_under_adiabatic_contraction():
    """classes.py:1488 -> cdm.py:203 -- AC_profiles takes no halo_type/gamma.

    Einasto + Cautun returns results byte-identical to NFW + Cautun.
    """
    nfw = CDM_profile(M200, CONC, q0=1.0, Phi_b=hernquist_phi,
                      halo_type="NFW", AC_prescription="Cautun")
    ein = CDM_profile(M200, CONC, q0=1.0, Phi_b=hernquist_phi,
                      halo_type="Einasto", gamma=0.40, AC_prescription="Cautun")

    rr = np.array([1.0, 5.0, 20.0, 100.0])
    rho_nfw = np.array([nfw.rho_sph(r) for r in rr])
    rho_ein = np.array([ein.rho_sph(r) for r in rr])

    assert not np.allclose(rho_nfw, rho_ein, rtol=1e-12), (
        "Einasto+Cautun is byte-identical to NFW+Cautun: halo_type is discarded"
    )
    # and the difference should be large, not marginal
    assert np.max(np.abs(rho_ein / rho_nfw - 1.0)) > 0.1


def test_d2_gamma_is_not_ignored_under_ac():
    """A scan over gamma with AC enabled must not produce a flat likelihood."""
    rr = np.array([1.0, 5.0, 20.0])
    out = []
    for g in (0.18, 0.30, 0.40):
        p = CDM_profile(M200, CONC, q0=1.0, Phi_b=hernquist_phi,
                        halo_type="Einasto", gamma=g, AC_prescription="Cautun")
        out.append(np.array([p.rho_sph(r) for r in rr]))
    spread = np.max([np.max(np.abs(out[i] / out[0] - 1.0)) for i in (1, 2)])
    assert spread > 1e-6, f"gamma has no effect under AC (spread {spread:.2e})"


# --------------------------------------------------------------------- D3
def test_d3_phi_accepts_array_theta():
    """classes.py:503 -- `theta == None` is an elementwise compare for arrays,
    and range(Lmax+1) walks odd L that were never solved for."""
    h = jeans.isothermal(R1, M200, CONC, q0=Q0, Phi_b=hernquist_phi,
                         L_list=[0, 2], r_grid=60)
    assert h is not None
    th = np.array([0.0, np.pi / 4, np.pi / 2])
    val = h.Phi(0.9 * R1, th)
    assert np.shape(val) == np.shape(th)
    assert np.all(np.isfinite(val))


def test_d3_phi_dm_does_not_pass_self_twice():
    """classes.py:512 -- recursive calls pass `self` as the first positional
    argument on top of the bound receiver, and line 530 uses an undefined
    `theta` instead of `th`."""
    h = jeans.isothermal(R1, M200, CONC, q0=Q0, Phi_b=hernquist_phi,
                         L_list=[0, 2], r_grid=60)
    assert h is not None
    val = h.Phi_dm(0.9 * R1, np.pi / 3, Lmax=2)
    assert np.all(np.isfinite(val))


# --------------------------------------------------------------------- D4
def test_d4_unsupported_phi_b_signature_gives_a_clear_error():
    """tools.py:111 -- `num_sph_coords` is undefined, so the guard raises
    NameError instead of the intended message."""
    def bad_phi_b(r, theta, extra):          # 3 parameters: unsupported
        return -GN * MSTAR / (r + A_H)

    with pytest.raises(Exception) as ei:
        jeans.tools.compute_Mb(bad_phi_b, 1e-3, 100.0)
    assert not isinstance(ei.value, NameError), (
        f"guard raised NameError instead of a readable message: {ei.value}"
    )
    assert "phi_b must take" in str(ei.value).lower()


# --------------------------------------------------------------------- D5
def test_d5_all_zero_M_list_is_accepted():
    """classes.py:1636 -- rejects M_list=[0,0,0] although every entry is zero."""
    outer = CDM_profile(M200, CONC, q0=Q0, halo_type="NFW")
    moments = outer.compute_potential_moments(R1, L_list=[0, 2, 4],
                                              M_list=[0, 0, 0])
    assert len(moments) == 3
    assert np.all(np.isfinite(moments))


# --------------------------------------------------------------------- D6
def test_d6_spherical_and_isothermal_use_the_same_baryon_average():
    """gen.py:26 vs :112 -- jeans.spherical passes the spherically averaged
    potential (m0 = exp(-<Phi_b>)) while jeans.isothermal passes the full
    Phi_b (m0 = <exp(-Phi_b)>). These differ by a Jensen gap.

    This test pins the *rigorous* convention for both.
    """
    # The gap is strongly parameter dependent; this is the configuration where
    # it was measured at -1.55% in r0 (r1=30, M200=1e13 shows only -0.01%).
    r1d, m200d, cd = 10.0, 1e12, 10.0
    sph = jeans.spherical(r1d, m200d, cd, Phi_b=mn_phi)
    iso = jeans.isothermal(r1d, m200d, cd, q0=1.0, Phi_b=mn_phi, L_list=[0])
    assert sph is not None and iso is not None

    rel_r0 = abs(sph.inner.r0 / iso.inner.r0 - 1.0)
    rel_s0 = abs(sph.inner.sigma0 / iso.inner.sigma0 - 1.0)
    assert rel_r0 < 1e-3, (
        f"spherical and isothermal(L=[0]) disagree by {rel_r0:.2%} in r0: "
        "they are averaging the baryon potential differently"
    )
    assert rel_s0 < 1e-3
