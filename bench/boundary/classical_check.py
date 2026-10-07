"""Is the existence boundary physics, or is it the solver?

Prong 1 of RESEARCH_PLAN.md rests entirely on R_fold being a real turning point
of the isothermal sequence. If it is a convergence artifact, every exclusion
statement built on it is a self-portrait. This is the external check, and it
goes through a textbook number the code was never fitted to.

The march is dphi/du = (u/3) exp(-eta), deta/du = -(3/u) expm1(eta - phi), with
rho/rho_0 = exp(-phi) and M = (4 pi/3) rho_0 r^3 exp(-eta). For an isothermal
sphere held at fixed external pressure P = rho_edge c_s^2, the dimensionless
mass m ~ M P^(1/2) is maximised at the Bonnor-Ebert critical contrast
rho_c/rho_edge = 14.04. That value is classical and independent of anything
here.

Result: 14.0420. Four significant figures, nothing tuned.

A note on what this does NOT show. An earlier plan called for recovering
Antonov's microcanonical contrast of 708.61 at OUR fold. That expectation was
wrong, and the check would have "failed" for the wrong reason. R = rho_bar /
(3 rho_edge) is its own functional and its maximum is a different turning point
of the same one-parameter sequence: we get 290.47, not 708.61. The sequence is
classical; which of its turning points R happens to fold at is a separate
question from whether the sequence is right.
"""
import numpy as np

from jeanie import universal as U

BONNOR_EBERT = 14.04          # classical, fixed external pressure
ANTONOV = 708.61              # classical, microcanonical -- NOT our fold


def report():
    u, phi, eta = U.table()
    g = phi - eta                       # log(3R)
    rho = np.exp(-phi)                  # rho/rho_0
    mbar = np.exp(-eta)                 # rho_bar/rho_0

    i_fold = int(np.argmax(g))
    # m ~ M P^{1/2} in units that do not move the maximum
    m_be = mbar * u ** 3 * np.sqrt(rho)
    i_be = int(np.argmax(m_be[1:])) + 1

    be = 1.0 / rho[i_be]
    fold = 1.0 / rho[i_fold]
    R_max = mbar[i_fold] / rho[i_fold] / 3.0

    print(f"{'quantity':40s} {'u':>10s} {'rho_c/rho_edge':>16s}")
    print(f"{'canonical turning point (Bonnor-Ebert)':40s} {u[i_be]:10.5f} "
          f"{be:16.4f}   classical {BONNOR_EBERT}")
    print(f"{'first fold of R (our criterion)':40s} {u[i_fold]:10.5f} "
          f"{fold:16.4f}   (Antonov is {ANTONOV}, a different turning point)")
    print(f"\nBonnor-Ebert agreement: {be/BONNOR_EBERT:.5f}  "
          f"-> the march IS the classical isothermal sphere")
    print(f"R_MAX from the table: {R_max:.10f}   module constant "
          f"{U.R_MAX:.10f}   ratio {R_max/U.R_MAX:.12f}")
    return be, fold, R_max


if __name__ == "__main__":
    report()
