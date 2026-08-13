"""Chapter-17 dimensionless coherent-field formulation.

For ``lambda > 0`` Volume 2 introduces

    phi = sqrt(lambda / mu) * psi,
    xi = 1 / sqrt(lambda),

and the static dimensionless equation

    -laplacian(psi) - psi + |psi|^2 psi = 0,

with effective potential ``W(psi) = (1/4)(1-|psi|^2)^2``.

These helpers implement that source rescaling without introducing new physical
parameters or a second quartic-potential implementation for ``phi``.

Scientific status: research.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from ..numerics import UniformRectilinearGrid, laplacian
from ..numerics.boundaries import Boundary

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _positive(value, *, name: str) -> float:
    try:
        scalar = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return scalar


def coherence_length(lambda_: float) -> float:
    """Return the Chapter-17 length scale ``xi = 1/sqrt(lambda)``."""

    lambda_value = _positive(lambda_, name="lambda_")
    return float(1.0 / np.sqrt(lambda_value))


def to_dimensionless_field(phi, *, lambda_: float, mu: float) -> np.ndarray:
    """Return ``psi = phi / sqrt(lambda/mu)`` from Eq. (17.1)."""

    lambda_value = _positive(lambda_, name="lambda_")
    mu_value = _positive(mu, name="mu")
    field = np.asarray(phi)
    if field.size == 0 or not np.issubdtype(field.dtype, np.number):
        raise ValueError("phi must be a non-empty real or complex numeric array")
    if not np.all(np.isfinite(field)):
        raise ValueError("phi must contain only finite values")
    return field / np.sqrt(lambda_value / mu_value)


def from_dimensionless_field(psi, *, lambda_: float, mu: float) -> np.ndarray:
    """Return ``phi = sqrt(lambda/mu) psi`` from Eq. (17.1)."""

    lambda_value = _positive(lambda_, name="lambda_")
    mu_value = _positive(mu, name="mu")
    field = np.asarray(psi)
    if field.size == 0 or not np.issubdtype(field.dtype, np.number):
        raise ValueError("psi must be a non-empty real or complex numeric array")
    if not np.all(np.isfinite(field)):
        raise ValueError("psi must contain only finite values")
    return np.sqrt(lambda_value / mu_value) * field


def dimensionless_effective_potential(psi) -> np.ndarray:
    """Return ``W(psi) = (1/4)(1-|psi|^2)^2`` from Chapter 17."""

    field = np.asarray(psi)
    if field.size == 0 or not np.issubdtype(field.dtype, np.number):
        raise ValueError("psi must be a non-empty real or complex numeric array")
    if not np.all(np.isfinite(field)):
        raise ValueError("psi must contain only finite values")
    return 0.25 * (1.0 - np.abs(field) ** 2) ** 2


def dimensionless_static_residual(
    psi,
    grid: UniformRectilinearGrid,
    *,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Evaluate the source Eq. (17.2) static residual.

    The grid coordinates are interpreted as the dimensionless coordinates of
    Chapter 17.  Spatial discretisation is delegated to the existing Numerics
    Laplacian.
    """

    field = np.asarray(psi)
    if field.shape != grid.shape:
        raise ValueError(f"psi shape {field.shape} does not match grid shape {grid.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError("psi must contain only finite values")
    return -laplacian(field, grid, boundary=boundary) - field + np.abs(field) ** 2 * field
