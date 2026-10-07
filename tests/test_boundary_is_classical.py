import numpy as np

from jeanie import universal as U

BONNOR_EBERT = 14.04


def test_march_reproduces_the_bonnor_ebert_critical_contrast():
    """The external check prong 1 rests on.

    If the isothermal march is right, the canonical turning point must sit at
    the classical Bonnor-Ebert contrast of 14.04 -- a number this code was
    never fitted to. Without this, R_fold could be a convergence artifact and
    every exclusion statement built on it would be circular.
    """
    u, phi, eta = U.table()
    rho, mbar = np.exp(-phi), np.exp(-eta)
    i = int(np.argmax((mbar * u ** 3 * np.sqrt(rho))[1:])) + 1
    assert abs((1.0 / rho[i]) / BONNOR_EBERT - 1.0) < 2e-3


def test_R_MAX_constant_matches_the_table():
    """R_MAX is derived, not asserted, so the two must agree exactly."""
    u, phi, eta = U.table()
    i = int(np.argmax(phi - eta))
    assert np.exp(phi[i] - eta[i]) / 3.0 == __import__("pytest").approx(
        U.R_MAX, rel=1e-9)
