"""Generic dynamical-systems utilities and legacy compatibility boundaries.

This namespace is not an authority for the canonical Theory of Agencity. The
canonical observable pipeline is owned by ``agencitylab.core`` and orchestrated
by :func:`agencitylab.compute_agencity`.

Generic attractor, bifurcation, delay, stability, and integration helpers are
retained where they have software value independent of Agencity physics. The
historical Agencity-specific RHS is retired and available only as an explicit
compatibility boundary that raises instead of executing scientifically
misleading equations.
"""

from __future__ import annotations

from importlib import import_module

from .attractors import detect_attractor_type
from .bifurcation import scan_bifurcation
from .delays import interpolate_history, solve_delay_euler
from .integrators import solve_euler, solve_ivp_wrapper
from .stability import is_bounded_trajectory, linear_stability_hint, lyapunov_like_indicator

_LEGACY_NAMES = {
    "AgencityState",
    "agencity_rhs",
    "beta_and_b_from_trajectory",
    "beta_from_state",
    "default_system_rhs",
    "rk4_step",
}


def __getattr__(name: str):
    if name == "rk4_step":
        return getattr(import_module("agencitylab.dynamics.integrators"), name)
    if name in _LEGACY_NAMES:
        return getattr(import_module("agencitylab.dynamics.system"), name)
    raise AttributeError(f"module 'agencitylab.dynamics' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LEGACY_NAMES)


__all__ = [
    "detect_attractor_type",
    "interpolate_history",
    "is_bounded_trajectory",
    "linear_stability_hint",
    "lyapunov_like_indicator",
    "scan_bifurcation",
    "solve_delay_euler",
    "solve_euler",
    "solve_ivp_wrapper",
]
