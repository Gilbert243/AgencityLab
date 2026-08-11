"""Informative CFL-style diagnostics for uniform-grid explicit schemes.

These helpers provide sufficient numerical guidelines for common centered
finite-difference discretisations.  They are not universal stability
guarantees and they do not enforce a time step.
"""

from __future__ import annotations

import numpy as np

from .grid import UniformRectilinearGrid


def _validate_nonnegative(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def wave_cfl_limit(grid: UniformRectilinearGrid, wave_speed: float) -> float:
    """Return a sufficient CFL time-step guideline for a centered wave scheme.

    The diagnostic is

    ``dt <= 1 / (c * sqrt(sum_i 1/h_i**2))``.

    A zero wave speed returns infinity because this particular wave CFL
    restriction vanishes.  The value is informative only; solver-specific
    stability can impose a stricter condition.
    """

    speed = _validate_nonnegative(wave_speed, "wave_speed")
    if speed == 0.0:
        return float("inf")
    inverse_squared_sum = sum(1.0 / (spacing * spacing) for spacing in grid.spacings)
    return 1.0 / (speed * np.sqrt(inverse_squared_sum))


def diffusion_cfl_limit(grid: UniformRectilinearGrid, diffusivity: float) -> float:
    """Return a sufficient explicit-diffusion time-step guideline.

    For the standard centered Laplacian with forward-Euler-type explicit
    stepping, the diagnostic is

    ``dt <= 1 / (2 * alpha * sum_i 1/h_i**2)``.

    Zero diffusivity returns infinity.  This is a numerical guideline, not a
    universal stability guarantee for arbitrary integrators or equations.
    """

    alpha = _validate_nonnegative(diffusivity, "diffusivity")
    if alpha == 0.0:
        return float("inf")
    inverse_squared_sum = sum(1.0 / (spacing * spacing) for spacing in grid.spacings)
    return 1.0 / (2.0 * alpha * inverse_squared_sum)
