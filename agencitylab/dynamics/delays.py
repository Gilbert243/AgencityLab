"""
Delay-differential utilities for AgencityLab.

The base implementation uses a simple history interpolation function and
an explicit Euler-like scheme for educational and prototyping purposes.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def interpolate_history(history_xi, history_y, xi_query):
    """
    Interpolate a history trajectory at arbitrary query coordinates.
    """
    history_xi = np.asarray(history_xi, dtype=float)
    history_y = np.asarray(history_y, dtype=float)
    xi_query = np.asarray(xi_query, dtype=float)

    if history_xi.ndim != 1:
        raise ValueError("history_xi must be one-dimensional.")
    if history_y.ndim == 1:
        return np.interp(xi_query, history_xi, history_y)

    if history_y.ndim != 2:
        raise ValueError("history_y must be one- or two-dimensional.")

    output = np.zeros((xi_query.size, history_y.shape[1]), dtype=float)
    for column in range(history_y.shape[1]):
        output[:, column] = np.interp(xi_query, history_xi, history_y[:, column])
    return output


def solve_delay_euler(
    rhs: Callable[[float, np.ndarray, Callable[[float], np.ndarray]], np.ndarray],
    history_function: Callable[[float], np.ndarray],
    xi_grid,
    delay: float,
    y0: np.ndarray,
):
    """
    Solve a delay differential equation with a simple explicit method.

    Parameters
    ----------
    rhs:
        Function of the form rhs(xi, y, y_delayed).
    history_function:
        Function returning the state for xi < xi0.
    xi_grid:
        Grid where the solution is evaluated.
    delay:
        Constant delay.
    y0:
        Initial state at the first grid point.
    """
    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional.")
    if delay < 0:
        raise ValueError("delay must be non-negative.")

    trajectory = np.zeros((xi_grid.size, y0.size), dtype=float)
    trajectory[0] = y0

    def state_at(query_xi: float) -> np.ndarray:
        if query_xi <= xi_grid[0]:
            return np.asarray(history_function(query_xi), dtype=float)
        idx = np.searchsorted(xi_grid[: len(trajectory)], query_xi, side="right") - 1
        idx = max(0, min(idx, len(trajectory) - 1))
        return trajectory[idx]

    for i in range(1, xi_grid.size):
        h = float(xi_grid[i] - xi_grid[i - 1])
        delayed_state = state_at(float(xi_grid[i - 1] - delay))
        derivative = np.asarray(rhs(xi_grid[i - 1], trajectory[i - 1], delayed_state), dtype=float)
        trajectory[i] = trajectory[i - 1] + h * derivative

    return trajectory
