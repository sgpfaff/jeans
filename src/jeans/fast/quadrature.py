"""Fixed-node angular quadrature, and tabulation of an arbitrary baryon potential.

The package evaluates its theta integrals with adaptive scipy.quad, which is
86-89% of the runtime of a two-dimensional build and calls the tesseral harmonic
routine well over a million times per solve. Nothing about the angular kernel
requires adaptivity: the integrands are smooth in cos(theta), so a fixed
Gauss-Legendre rule with the harmonics tabulated once is both faster and more
accurate. Measured convergence is 5e-15 at 12 nodes against a 256-node
reference, and 10, 12, 16 and 20 nodes are indistinguishable in the final
density.

Fixing the nodes also matters for differentiability: an adaptive rule changes
its node count as parameters move, which puts small discontinuities into the
output and is one of the two reasons the package's own finite-difference
gradients lose their plateau.
"""
import numpy as np
from scipy.special import eval_legendre

__all__ = ["gauss_legendre", "harmonics", "tabulate_baryons", "N_GL_DEFAULT"]

N_GL_DEFAULT = 16

_CACHE = {}


def gauss_legendre(n=N_GL_DEFAULT, half_range=True):
    """Nodes and weights in x = cos(theta), normalised so the weights sum to 1.

    half_range=True integrates over x in [0, 1] and relies on mirror symmetry
    about the equatorial plane: all L even and Phi_b symmetric in z. That holds
    for a Miyamoto-Nagai disc or any spheroid centred on the origin, and it is
    what lets 16 nodes reach 1e-13. A warped, offset or rotating baryon
    distribution breaks it and needs half_range=False.
    """
    key = ("gl", n, half_range)
    if key in _CACHE:
        return _CACHE[key]
    x, w = np.polynomial.legendre.leggauss(n)
    if half_range:
        x = 0.5 * (x + 1.0)
        w = 0.5 * w
    else:
        w = 0.5 * w
    _CACHE[key] = (x, w)
    return x, w


def harmonics(L_list, n=N_GL_DEFAULT, half_range=True):
    """Z_L(x_j) on the quadrature nodes, for M = 0. Shape (len(L_list), n).

    Z(L, 0, theta, 0) = sqrt((2L+1)/4pi) P_L(cos theta), with the Condon-Shortley
    phase equal to 1 for M = 0.
    """
    key = ("Z", tuple(L_list), n, half_range)
    if key in _CACHE:
        return _CACHE[key]
    x, _ = gauss_legendre(n, half_range)
    Z = np.array(
        [np.sqrt((2 * L + 1) / (4.0 * np.pi)) * eval_legendre(L, x) for L in L_list]
    )
    _CACHE[key] = Z
    return Z


def tabulate_baryons(Phi_b, r_nodes, r_half, n=N_GL_DEFAULT, half_range=True):
    """Tabulate dPhi_b(r, theta) = Phi_b(r, theta) - Phi_b(0) on the solver nodes.

    Returns (grid_nodes, grid_half, weights), the grids having shape
    (len(r), n_theta). Independent of sigma0, so this is computed once per solve
    and reused across every Newton iterate and every continuation stage.

    Accepts a one-argument Phi_b(r) as well as the two-argument Phi_b(r, theta);
    a one-argument potential is broadcast across theta, which costs one extra
    axis of memory and keeps a single code path in the solver. For a spherically
    symmetric Phi_b the angular average is exact for any node count, since the
    integrand is constant in theta.
    """
    from inspect import signature

    n_args = len(signature(Phi_b).parameters)
    if n_args not in (1, 2):
        raise ValueError(
            "Phi_b must take 1 or 2 arguments, (r) or (r, theta); got %d." % n_args
        )

    x, w = gauss_legendre(n, half_range)
    theta = np.arccos(np.clip(x, -1.0, 1.0))

    if n_args == 1:
        phi0 = float(Phi_b(0.0))

        def grid_of(r):
            col = np.asarray([Phi_b(float(ri)) for ri in r], float) - phi0
            return np.repeat(col[:, None], len(x), axis=1)

    else:
        phi0 = float(Phi_b(0.0, np.pi / 2))

        def grid_of(r):
            R, T = np.meshgrid(np.asarray(r, float), theta, indexing="ij")
            return np.asarray(Phi_b(R, T), float) - phi0

    return grid_of(r_nodes), grid_of(r_half), w


def angular_average_exp(grid, weights, sig0sq):
    """< exp(-dPhi_b / sigma0^2) >_theta, evaluated stably. Reference for tests.

    The solver uses the numba version in kernels.source_from_grid; this is the
    plain-numpy equivalent, kept so the two can be cross-checked.
    """
    v = grid / sig0sq
    vmin = v.min(axis=1, keepdims=True)
    return np.exp(-vmin[:, 0]) * (weights * np.exp(-(v - vmin))).sum(axis=1)
