"""
Numerical integrators for AgencityLab dynamics.

The base layer provides a minimal Euler solver, a classic RK4 step and a
thin wrapper around scipy.integrate.solve_ivp when SciPy is available.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def solve_euler(rhs: Callable[[float, np.ndarray], np.ndarray], y0, xi_grid):
    """
    Integrate an ODE with the explicit Euler method.
    """
    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional.")

    trajectory = np.zeros((xi_grid.size, y0.size), dtype=float)
    trajectory[0] = y0

    for i in range(1, xi_grid.size):
        h = float(xi_grid[i] - xi_grid[i - 1])
        trajectory[i] = trajectory[i - 1] + h * np.asarray(rhs(xi_grid[i - 1], trajectory[i - 1]), dtype=float)

    return trajectory


def rk4_step(rhs: Callable[[float, np.ndarray], np.ndarray], xi: float, y: np.ndarray, h: float) -> np.ndarray:
    """
    Perform a single Runge-Kutta 4 step.
    """
    y = np.asarray(y, dtype=float)

    k1 = np.asarray(rhs(xi, y), dtype=float)
    k2 = np.asarray(rhs(xi + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(rhs(xi + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(rhs(xi + h, y + h * k3), dtype=float)

    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ivp_wrapper(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0,
    xi_grid,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """
    Integrate an ODE using scipy.integrate.solve_ivp if SciPy is installed.
    Falls back to Euler when SciPy is not available.
    """
    try:
        from scipy.integrate import solve_ivp  # type: ignore
    except Exception:
        return solve_euler(rhs, y0, xi_grid)

    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional.")

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y),
        t_span=(float(xi_grid[0]), float(xi_grid[-1])),
        y0=y0,
        t_eval=xi_grid,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.y.T
