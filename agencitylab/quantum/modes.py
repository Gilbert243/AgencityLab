"""Mode spectrum for the speculative quantised Agencity field.

Scientific source: *Agencity — Advanced Mathematical Foundations and Extensions*,
Volume 2, Chapter 21. The functions in this module implement only the explicit
broken-symmetry masses and dispersion relations stated there. They do not
quantise the canonical observable pipeline ``u -> beta -> b``.
"""

from __future__ import annotations

import numpy as np

from agencitylab.fields.physics import QuarticAgencityPotential, vacuum_amplitude
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


def _finite_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_wave_number(k_magnitude) -> np.ndarray:
    k = np.asarray(k_magnitude, dtype=float)
    if not np.all(np.isfinite(k)):
        raise ValueError("k_magnitude must contain only finite values")
    if np.any(k < 0.0):
        raise ValueError("k_magnitude must be non-negative")
    return k


def radial_mass_squared(lambda_) -> float:
    """Return the Chapter-21 radial-mode mass squared ``m_h^2 = 2 lambda``.

    The broken-symmetry expansion exists only for ``lambda > 0``.
    """
    lambda_value = _finite_scalar(lambda_, name="lambda_")
    if lambda_value <= 0.0:
        raise ValueError("broken-symmetry radial mode requires lambda_ > 0")
    return 2.0 * lambda_value


def radial_mass(lambda_) -> float:
    """Return the positive radial-mode mass ``sqrt(2 lambda)``."""
    return float(np.sqrt(radial_mass_squared(lambda_)))


def goldstone_mass_squared() -> float:
    """Return the Chapter-21 Goldstone-mode mass squared, exactly zero."""
    return 0.0


def radial_angular_frequency(k_magnitude, lambda_) -> np.ndarray:
    """Return ``omega_h(k) = sqrt(k^2 + 2 lambda)``."""
    k = _nonnegative_wave_number(k_magnitude)
    return np.sqrt(k**2 + radial_mass_squared(lambda_))


def goldstone_angular_frequency(k_magnitude) -> np.ndarray:
    """Return the massless Goldstone dispersion ``omega_theta(k) = |k|``."""
    return _nonnegative_wave_number(k_magnitude).copy()


def broken_symmetry_vacuum_amplitude(potential: QuarticAgencityPotential) -> float:
    """Return ``sqrt(lambda/mu)`` using the shared classical field potential.

    The potential is not duplicated in the quantum layer. Its existing
    validation and convention remain authoritative.
    """
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    return vacuum_amplitude(potential.lambda_, potential.mu)
