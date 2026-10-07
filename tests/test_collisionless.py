import numpy as np
import pytest

from jeanie.collisionless import einasto_seed, eddington_f, rho_of_psi

SEED = dict(rho_m2=1.0e7, r_m2=20.0, alpha=0.18)


@pytest.fixture(scope="module")
def inverted():
    r, rho, psi, drho, M = einasto_seed(**SEED)
    E, f = eddington_f(psi, drho)
    return r, rho, psi, E, f


def test_distribution_function_is_positive(inverted):
    """A negative f(E) means the seed admits no isotropic DF at all.

    This is the first thing to fail if the energy grid stops resolving the top
    of the range: with an NFW seed and a plain geomspace grid we measured
    min f = -345.
    """
    _, _, _, E, f = inverted
    assert np.all(f > 0.0)


def test_round_trip_recovers_the_seed_density(inverted):
    """Re-integrating f in the ORIGINAL potential must give the seed back.

    Over 0.01 to 100 scale radii, which brackets every radius where baryons
    matter. Without this the shape response computed later is unfalsifiable.
    """
    r, rho, psi, E, f = inverted
    m = (r > 0.01 * SEED["r_m2"]) & (r < 100.0 * SEED["r_m2"])
    err = np.abs(rho_of_psi(psi[m], E, f) / rho[m] - 1.0)
    assert np.median(err) < 1e-3
    assert err.max() < 5e-3


def test_density_rises_monotonically_with_depth(inverted):
    """Deepening the potential must raise the density -- that IS contraction.

    Not imposed by a prescription: it falls out of integrating a fixed f over
    a deeper well.
    """
    _, _, psi, E, f = inverted
    ref = rho_of_psi(psi[::40], E, f)
    prev = ref
    for fac in (1.05, 1.2, 1.5):
        cur = rho_of_psi(psi[::40] * fac, E, f)
        assert np.all(cur >= prev * (1.0 - 1e-9))
        prev = cur


def test_potential_deeper_than_the_seed_is_finite(inverted):
    """F(Psi) must stay finite above the seed's own Psi_max.

    This is the whole reason for going through the DF rather than fitting
    rho(Psi) to the spherical pairs: with baryons the total potential runs
    deeper than the seed ever does, and a fitted F has nothing to say there
    while the velocity integral simply exhausts the support of f.
    """
    _, _, psi, E, f = inverted
    out = rho_of_psi(np.array([1.2, 2.0, 5.0]) * psi.max(), E, f)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)
    assert np.all(np.diff(out) > 0.0)


def test_seed_potential_matches_the_enclosed_mass(inverted):
    """dPsi/dr = -G M(r)/r^2, so the quadrature for Psi and the one for M
    have to agree. They are computed by different integrals."""
    r, rho, psi, _, _ = inverted
    _, _, _, _, M = einasto_seed(**SEED)
    m = (r > 0.05 * SEED["r_m2"]) & (r < 50.0 * SEED["r_m2"])
    num = np.gradient(psi, r)[m]
    exact = -4.302e-6 * M[m] / r[m] ** 2
    assert np.median(np.abs(num / exact - 1.0)) < 2e-3


@pytest.mark.xfail(reason="known open issue, diagnosed not guessed: for an "
                          "Einasto seed Psi_max - Psi ~ r^2 while rho_0 - rho "
                          "~ r^alpha, so drho/dPsi ~ (Psi_max-Psi)^(alpha/2-1) "
                          "= ^(-0.91) at alpha=0.18 -- an integrable but real "
                          "divergence. The velocity integrand then carries a "
                          "spike no fixed quadrature resolves, and F(Psi) is "
                          "discontinuous at the 0.5-5x level across Psi_max. "
                          "TWO fixes tried and REJECTED, both for the same "
                          "reason: at alpha=0.18 every central limit is "
                          "numerically unreachable. (1) Replacing f by its "
                          "asymptotic form f ~ tau^((alpha-3)/2): measured "
                          "exponents -2.03/-1.79/-1.61/-1.39 against "
                          "-1.44/-1.41/-1.375/-1.30 for alpha="
                          "0.12/0.18/0.25/0.40, since the correction is "
                          "O(tau^(alpha/2)) and needs tau/Psi_max ~ 1e-11. "
                          "(2) Subtracting the singular part with the "
                          "analytic coefficient: the expansion "
                          "rho_0 - rho ~ rho_0 (2/alpha)(r/r_m2)^alpha needs "
                          "(2/alpha) x^alpha << 1, which is 1.4 at x=1e-5, so "
                          "the analytic rho_0 overshoots the measured central "
                          "density by 4x and the subtraction made the round "
                          "trip worse (2.6e-2 vs 1.6e-4) with f going "
                          "negative. Affects only Psi > Psi_max, i.e. the "
                          "innermost sub-kpc; the shape response works "
                          "outside that, so the 2D solve proceeds with a "
                          "documented inner floor rather than blocking.",
                   strict=True)
def test_F_is_continuous_across_the_seed_potential_maximum(inverted):
    """F must not jump at Psi = max(E): the two regimes are the same integral.

    Marked xfail deliberately. The round-trip tests above cover 0.01-100 scale
    radii at the 1e-4 level, which is the regime the spherical model uses; this
    one covers the regime a deepened potential reaches, and it does not pass
    yet. Deleting it would hide the one place the construction is not ready.
    """
    _, _, _, E, f = inverted
    Em = E[-1]
    for d in (1e-3, 1e-4):
        a = rho_of_psi([Em * (1 - d)], E, f)[0]
        b = rho_of_psi([Em * (1 + d)], E, f)[0]
        assert abs(b / a - 1.0) < 0.02
