"""Momentum-space propagators stated for the speculative Agencity field.

The ``i epsilon`` parameter is an explicit numerical regulator supplied by the
caller. No fixed machine epsilon is inserted into the physical denominator.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from .modes import radial_mass_squared

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


def _finite_real_array(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite real values")
    return arr


def _positive_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def radial_propagator(k_squared, lambda_, *, epsilon) -> np.ndarray:
    """Return ``i / (k^2 - m_h^2 + i epsilon)`` with ``m_h^2 = 2 lambda``."""
    invariant = _finite_real_array(k_squared, name="k_squared")
    regulator = _positive_scalar(epsilon, name="epsilon")
    denominator = invariant - radial_mass_squared(lambda_) + 1j * regulator
    return 1j / denominator


def goldstone_propagator(k_squared, *, epsilon) -> np.ndarray:
    """Return the massless propagator ``i / (k^2 + i epsilon)``."""
    invariant = _finite_real_array(k_squared, name="k_squared")
    regulator = _positive_scalar(epsilon, name="epsilon")
    return 1j / (invariant + 1j * regulator)
