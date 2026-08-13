"""Generic ODE integration helpers retained for compatibility.

The authoritative fixed-step RK4 primitive is
``agencitylab.fields.numerics.rk4_step``. This module keeps the historical
``rk4_step`` location as a deprecated forwarding wrapper and retains the unique
Euler and optional-SciPy convenience solvers.
"""

from __future__ import annotations

from typing import Callable
import warnings

import numpy as np


def solve_euler(rhs: Callable[[float, np.ndarray], np.ndarray], y0, xi_grid):
    """Integrate an ODE with the explicit Euler method."""

    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional.")
    if xi_grid.size == 0:
        raise ValueError("xi_grid must not be empty.")

    trajectory = np.zeros((xi_grid.size, y0.size), dtype=float)
    trajectory[0] = y0
    for i in range(1, xi_grid.size):
        h = float(xi_grid[i] - xi_grid[i - 1])
        trajectory[i] = trajectory[i - 1] + h * np.asarray(
            rhs(xi_grid[i - 1], trajectory[i - 1]), dtype=float
        )
    return trajectory


def rk4_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    xi: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """Deprecated forwarding alias to the authoritative generic RK4 primitive."""

    warnings.warn(
        "agencitylab.dynamics.rk4_step is deprecated; use "
        "agencitylab.fields.numerics.rk4_step.",
        DeprecationWarning,
        stacklevel=2,
    )
    from agencitylab.fields.numerics.integrators import rk4_step as authoritative_rk4_step

    return authoritative_rk4_step(rhs, xi, y, h)


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
        from scipy.integrate import solve_ivp  # type: ignore
    except Exception:
        return solve_euler(rhs, y0, xi_grid)

    xi_grid = np.asarray(xi_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    if xi_grid.ndim != 1:
        raise ValueError("xi_grid must be one-dimensional.")
    if xi_grid.size == 0:
        raise ValueError("xi_grid must not be empty.")

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
