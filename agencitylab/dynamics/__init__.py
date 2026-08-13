"""Generic dynamical-systems utilities.

This namespace contains reusable numerical helpers only. It is not an authority
for the canonical Theory of Agencity, which is implemented in
:mod:`agencitylab.core` and orchestrated by :func:`agencitylab.compute_agencity`.
"""

from .attractors import detect_attractor_type
from .bifurcation import scan_bifurcation
from .delays import interpolate_history, solve_delay_euler
from .integrators import solve_euler, solve_ivp_wrapper
from .stability import is_bounded_trajectory, linear_stability_hint, lyapunov_like_indicator

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
