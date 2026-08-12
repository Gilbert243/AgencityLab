"""Quartic research potential for the autonomous Agencity field.

Scientific source: *Agencity — Advanced Mathematical Foundations and Extensions*,
Volume 2. In dimensionless/natural units,
``V(phi) = -(lambda/2)|phi|^2 + (mu/4)|phi|^4`` with ``mu > 0``.

The autonomous ``phi`` field and this potential have scientific status
``research`` and no empirical validation is claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agencitylab.models.field_extensions import ParameterProvenance
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_field(phi) -> np.ndarray:
    arr = np.asarray(phi)
    if not np.all(np.isfinite(arr)):
        raise ValueError("phi must contain only finite values")
    return arr


@dataclass(frozen=True, slots=True)
class QuarticAgencityPotential:
    """Volume-2 quartic potential in dimensionless/natural units.

    ``lambda_`` may have either sign but must be finite. ``mu`` must be finite
    and strictly positive. The optional provenance fields reuse AgencityLab's
    shared parameter-provenance contract rather than defining a second system.

    ``gradient(phi)`` implements the complex Wirtinger derivative
    ``dV/d(phi*) = -lambda*phi + mu*|phi|^2*phi``. For real ``phi`` the same
    expression applies naturally; no separate real-field physics is introduced.
    """

    lambda_: float
    mu: float
    lambda_provenance: ParameterProvenance | None = None
    mu_provenance: ParameterProvenance | None = None
    scientific_status: ScientificStatus = ScientificStatus.RESEARCH
    units_convention: str = "dimensionless"

    def __post_init__(self) -> None:
        lambda_value = _finite_scalar(self.lambda_, name="lambda_")
        mu_value = _finite_scalar(self.mu, name="mu")
        if mu_value <= 0.0:
            raise ValueError("mu must be strictly positive")
        if self.scientific_status is not ScientificStatus.RESEARCH:
            raise ValueError("quartic field potential must have scientific status 'research'")
        if self.units_convention not in {"dimensionless", "natural_units"}:
            raise ValueError("units_convention must be 'dimensionless' or 'natural_units'")
        for item, name in (
            (self.lambda_provenance, "lambda_provenance"),
            (self.mu_provenance, "mu_provenance"),
        ):
            if item is not None and not isinstance(item, ParameterProvenance):
                raise TypeError(f"{name} must be ParameterProvenance or None")
        object.__setattr__(self, "lambda_", lambda_value)
        object.__setattr__(self, "mu", mu_value)

    def value(self, phi) -> np.ndarray:
        """Return the real quartic potential density ``V(phi)``."""
        arr = _finite_field(phi)
        modulus_squared = np.abs(arr) ** 2
        return -(self.lambda_ / 2.0) * modulus_squared + (self.mu / 4.0) * modulus_squared**2

    def gradient(self, phi) -> np.ndarray:
        """Return ``dV/d(phi*)`` under the Volume-2 complex convention."""
        arr = _finite_field(phi)
        return -self.lambda_ * arr + self.mu * (np.abs(arr) ** 2) * arr


def vacuum_amplitude(lambda_, mu) -> float:
    """Return ``sqrt(lambda/mu)`` for the broken-symmetry vacuum.

    A non-zero vacuum manifold exists only for finite ``lambda > 0`` and
    finite ``mu > 0``. ``lambda <= 0`` is valid for the potential itself but
    does not define a non-zero broken vacuum and is rejected here explicitly.
    """
    lambda_value = _finite_scalar(lambda_, name="lambda_")
    mu_value = _finite_scalar(mu, name="mu")
    if mu_value <= 0.0:
        raise ValueError("mu must be strictly positive")
    if lambda_value <= 0.0:
        raise ValueError("non-zero vacuum requires lambda_ > 0")
    return float(np.sqrt(lambda_value / mu_value))


def vacuum_state(lambda_, mu, *, theta) -> complex:
    """Construct one caller-selected point on the U(1) vacuum manifold.

    No phase is privileged: ``theta`` is required explicitly rather than being
    silently defaulted to zero.
    """
    theta_value = _finite_scalar(theta, name="theta")
    amplitude = vacuum_amplitude(lambda_, mu)
    return complex(amplitude * np.exp(1j * theta_value))
