"""Dissipation and entropy-production evaluators for Agencity fields.

Scientific status: research.  These functions implement the local relations from
Volume 2, Chapter 18 without modifying the canonical ``u -> beta -> b`` pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np

from agencitylab.fields.numerics import UniformRectilinearGrid, integrate_spatial
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_numeric_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain real or complex numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_real_array(value, *, name: str) -> np.ndarray:
    array = _finite_numeric_array(value, name=name)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    return np.asarray(array, dtype=float)


def _finite_nonnegative_scalar(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def dissipation_density(phi_dot, gamma: float) -> np.ndarray:
    """Return ``Gamma * |partial_t phi|**2`` from Volume 2, Chapter 18.

    ``gamma`` must be finite and non-negative.  The exact ``gamma == 0``
    branch returns zero without epsilon regularisation.  Real and complex
    finite fields are supported and inputs are never mutated.
    """

    velocity = _finite_numeric_array(phi_dot, name="phi_dot")
    gamma_value = _finite_nonnegative_scalar(gamma, name="gamma")
    magnitude_squared = np.asarray(np.abs(velocity) ** 2, dtype=float)
    if gamma_value == 0.0:
        return np.zeros_like(magnitude_squared, dtype=float)
    return gamma_value * magnitude_squared


def entropy_production_density(phi_dot, gamma: float, t_eff) -> np.ndarray:
    """Return ``sigma = Gamma / T_eff * |partial_t phi|**2``.

    ``T_eff`` is an explicit physical/contextual input and must be finite and
    strictly positive everywhere.  No epsilon is inserted in the denominator.
    """

    q_dot = dissipation_density(phi_dot, gamma)
    temperature = _finite_real_array(t_eff, name="t_eff")
    if np.any(temperature <= 0.0):
        raise ValueError("t_eff must be strictly positive")
    try:
        q_broadcast, t_broadcast = np.broadcast_arrays(q_dot, temperature)
    except ValueError as exc:
        raise ValueError("t_eff is not broadcast-compatible with phi_dot") from exc
    return np.asarray(q_broadcast / t_broadcast, dtype=float)


def total_dissipated_power(
    phi_dot,
    gamma: float,
    grid: UniformRectilinearGrid,
) -> float:
    """Integrate the dissipated-power density on an existing numerical grid."""

    density = dissipation_density(phi_dot, gamma)
    return float(integrate_spatial(density, grid))


def total_entropy_production(
    phi_dot,
    gamma: float,
    t_eff,
    grid: UniformRectilinearGrid,
) -> float:
    """Integrate the local entropy-production density on an existing grid."""

    density = entropy_production_density(phi_dot, gamma, t_eff)
    return float(integrate_spatial(density, grid))


def dissipation_rate(energy_in, energy_out):
    """Legacy clipped energy-difference placeholder.

    This historical helper is not the Chapter-18 dissipation law and is kept
    only for compatibility.  New code should use :func:`dissipation_density`.
    """

    warnings.warn(
        "dissipation_rate is a legacy heuristic, not the Agencity field "
        "dissipation law; use dissipation_density instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return max(0.0, energy_in - energy_out)
