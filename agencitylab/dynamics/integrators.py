"""Generic ODE integration helpers.

The reusable fixed-step RK4 primitive lives in
:mod:`agencitylab.fields.numerics`; this module provides only the distinct Euler
and optional-SciPy convenience solvers.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def solve_euler(rhs: Callable[[float, np.ndarray], np.ndarray], y0, xi_grid):
    """Integrate an ODE with the explicit Euler method."""
    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional")
    if xi_grid.size == 0:
        raise ValueError("xi_grid must not be empty")

    trajectory = np.zeros((xi_grid.size, y0.size), dtype=float)
    trajectory[0] = y0
    for i in range(1, xi_grid.size):
        h = float(xi_grid[i] - xi_grid[i - 1])
        trajectory[i] = trajectory[i - 1] + h * np.asarray(
            rhs(xi_grid[i - 1], trajectory[i - 1]), dtype=float
        )
    return trajectory


def solve_ivp_wrapper(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0,
    xi_grid,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """Use SciPy ``solve_ivp`` when installed, otherwise fall back to Euler."""
    try:
        from scipy.integrate import solve_ivp  # type: ignore[import-not-found]
    except ImportError:
        return solve_euler(rhs, y0, xi_grid)

    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional")
    if xi_grid.size == 0:
        raise ValueError("xi_grid must not be empty")

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
