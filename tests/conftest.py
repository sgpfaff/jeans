"""Shared fixtures.

Every correctness test here compares against the package, and a package build
costs 0.3 s (spherical) to 30 s (isothermal at L=[0,2,4] with a disc). The same
configurations are needed by several tests, so without caching the suite spends
almost all its time rebuilding references it already has.

The caches are keyed on the full argument tuple and live for the session. They
hold package output only; nothing from the fast solvers is memoised, so a test
can never pass by comparing a cached result against itself.
"""
import numpy as np
import pytest

import jeans
from jeans.classes import CDM_profile
from jeans.definitions import GN

_PKG = {}
_OUTER = {}
_BOUNDARY = {}


def _key(*args):
    return tuple(
        (a.__name__ if callable(a) else tuple(a) if isinstance(a, (list, tuple)) else a)
        for a in args
    )


@pytest.fixture(scope="session")
def pkg_spherical():
    """Memoised jeans.spherical(...)."""
    def build(r1, M200, c, Phi_b=None, **kw):
        k = _key("sph", r1, M200, c, Phi_b, tuple(sorted(kw.items())))
        if k not in _PKG:
            _PKG[k] = jeans.spherical(r1, M200, c, Phi_b=Phi_b, **kw)
        return _PKG[k]
    return build


@pytest.fixture(scope="session")
def pkg_isothermal():
    """Memoised jeans.isothermal(...)."""
    def build(r1, M200, c, q0=1.0, Phi_b=None, L_list=(0,), **kw):
        k = _key("iso", r1, M200, c, q0, Phi_b, L_list, tuple(sorted(kw.items())))
        if k not in _PKG:
            _PKG[k] = jeans.isothermal(r1, M200, c, q0=q0, Phi_b=Phi_b,
                                       L_list=list(L_list), **kw)
        return _PKG[k]
    return build


@pytest.fixture(scope="session")
def outer_data():
    """Memoised (rho1, M1, J_L) from a CDM_profile at r1."""
    def build(r1, M200, c, q0=1.0, Phi_b=None, L_list=(0,)):
        k = _key("bd", r1, M200, c, q0, Phi_b, L_list)
        if k not in _BOUNDARY:
            ok = _key("cdm", M200, c, q0, Phi_b)
            if ok not in _OUTER:
                _OUTER[ok] = CDM_profile(M200, c, q0=q0, Phi_b=Phi_b)
            o = _OUTER[ok]
            _BOUNDARY[k] = (
                o.rho_sph_avg(r1),
                o.M_encl(r1),
                o.compute_potential_moments(r1, L_list=list(L_list),
                                            M_list=[0] * len(L_list)),
            )
        return _BOUNDARY[k]
    return build
