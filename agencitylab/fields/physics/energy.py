"""Pure energy primitives for the research Agencity field physics layer.

The functions implement the natural/dimensionless-unit decomposition
``rho = 1/2 |phi_dot|^2 + 1/2 |grad phi|^2 + V(phi)``. Spatial derivatives,
grids, and quadrature geometry are intentionally external inputs; this module
has no dependency on ``agencitylab.fields.numerics`` and no PDE solver.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from .potential import QuarticAgencityPotential

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_array(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _real_array(value, *, name: str) -> np.ndarray:
    arr = _finite_array(value, name=name)
    arr = np.real_if_close(arr, tol=1000)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be theoretically real")
    return np.asarray(arr, dtype=float)


def kinetic_energy_density(phi_dot) -> np.ndarray:
    """Return ``1/2 * |phi_dot|^2`` as a real finite density."""
    arr = _finite_array(phi_dot, name="phi_dot")
    return np.asarray(0.5 * np.abs(arr) ** 2, dtype=float)


def gradient_energy_density(gradient_norm_squared) -> np.ndarray:
    """Return ``1/2 * |grad phi|^2`` from a precomputed squared norm.

    The input must already represent the real non-negative squared norm of the
    spatial gradient. Computing that gradient belongs to the numerical layer.
    """
    norm_squared = _real_array(gradient_norm_squared, name="gradient_norm_squared")
    if np.any(norm_squared < 0.0):
        raise ValueError("gradient_norm_squared must be non-negative")
    return 0.5 * norm_squared


def potential_energy_density(phi, potential: QuarticAgencityPotential) -> np.ndarray:
    """Return the real quartic potential density for ``phi``."""
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    return _real_array(potential.value(phi), name="potential energy density")


def field_energy_density(
    phi,
    phi_dot,
    gradient_norm_squared,
    potential: QuarticAgencityPotential,
) -> np.ndarray:
    """Return the total real field-energy density in natural/dimensionless units."""
    kinetic = kinetic_energy_density(phi_dot)
    gradient = gradient_energy_density(gradient_norm_squared)
    potential_density = potential_energy_density(phi, potential)
    try:
        shape = np.broadcast_shapes(kinetic.shape, gradient.shape, potential_density.shape)
    except ValueError as exc:
        raise ValueError("energy-density components have incompatible shapes") from exc
    return (
        np.broadcast_to(kinetic, shape)
        + np.broadcast_to(gradient, shape)
        + np.broadcast_to(potential_density, shape)
    )


def total_field_energy(density, *, volume_element) -> float:
    """Integrate a supplied real density using explicit scalar or array weights.

    ``volume_element`` may be a non-negative finite scalar or an array with the
    exact same shape as ``density``. No grid, geometry, or volume convention is
    inferred implicitly.
    """
    rho = _real_array(density, name="density")
    weights = _real_array(volume_element, name="volume_element")
    if weights.ndim == 0:
        scalar = float(weights)
        if scalar < 0.0:
            raise ValueError("volume_element must be non-negative")
        return float(np.sum(rho) * scalar)
    if weights.shape != rho.shape:
        raise ValueError("array volume_element must have the exact density shape")
    if np.any(weights < 0.0):
        raise ValueError("volume_element weights must be non-negative")
    return float(np.sum(rho * weights))
