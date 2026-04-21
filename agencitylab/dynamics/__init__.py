"""
Dynamical system layer for AgencityLab.

This package contains the equations, integrators and qualitative analysis
utilities used to study the evolution of the Agencity variables.
"""

from .attractors import detect_attractor_type
from .bifurcation import scan_bifurcation
from .delays import interpolate_history, solve_delay_euler
from .integrators import rk4_step, solve_euler, solve_ivp_wrapper
from .stability import (
    lyapunov_like_indicator,
    is_bounded_trajectory,
    linear_stability_hint,
)
from .system import AgencityState, agencity_rhs, default_system_rhs

__all__ = [
    "AgencityState",
    "agencity_rhs",
    "default_system_rhs",
    "detect_attractor_type",
    "interpolate_history",
    "is_bounded_trajectory",
    "linear_stability_hint",
    "lyapunov_like_indicator",
    "rk4_step",
    "scan_bifurcation",
    "solve_delay_euler",
    "solve_euler",
    "solve_ivp_wrapper",
]
