"""
Bifurcation scan helpers for AgencityLab.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np

from .integrators import solve_ivp_wrapper


def scan_bifurcation(
    rhs_factory: Callable[[float], Callable[[float, np.ndarray], np.ndarray]],
    parameter_values,
    y0,
    xi_grid,
    observable_index: int = 0,
):
    """
    Scan a one-parameter family of dynamical systems.

    Parameters
    ----------
    rhs_factory:
        Callable returning rhs(xi, y) for a given parameter value.
    parameter_values:
        Iterable of parameter values.
    y0:
        Initial state.
    xi_grid:
        Integration grid.
    observable_index:
        Component extracted from the asymptotic regime.

    Returns
    -------
    dict
        Dictionary with parameter values and final observable values.
    """
    parameter_values = np.asarray(parameter_values, dtype=float)
    final_values = np.zeros(parameter_values.size, dtype=float)

    for i, param in enumerate(parameter_values):
        rhs = rhs_factory(float(param))
        trajectory = solve_ivp_wrapper(rhs, y0, xi_grid)
        final_values[i] = float(trajectory[-1, observable_index])

    return {
        "parameter_values": parameter_values,
        "final_values": final_values,
    }
