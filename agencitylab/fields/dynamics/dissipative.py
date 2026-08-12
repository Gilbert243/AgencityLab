"""Dissipative classical dynamics for the autonomous Agencity field.

This module adds only the Volume-2 friction term to the conservative field
primitive.  The spatial operator and quartic source remain owned by the
existing Numerics and Physics layers.

Scientific status: research. No empirical validation is claimed.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from ..numerics import UniformRectilinearGrid
from ..numerics.boundaries import Boundary
from ..physics import QuarticAgencityPotential
from .klein_gordon import klein_gordon_acceleration

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_gamma(gamma: float) -> float:
    """Return a finite non-negative friction coefficient."""

    try:
        value = float(gamma)
    except Exception as exc:
        raise ValueError("gamma must be a finite real scalar") from exc
    if not np.isfinite(value):
        raise ValueError("gamma must be finite")
    if value < 0.0:
        raise ValueError("gamma must be non-negative")
    return value


def dissipative_klein_gordon_acceleration(
    phi: np.ndarray,
    phi_dot: np.ndarray,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    gamma: float,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Return dissipative Klein-Gordon acceleration.

    The implemented research equation is the conservative acceleration minus
    ``gamma * phi_dot``.  ``gamma`` must be finite and non-negative.  The exact
    ``gamma == 0`` branch returns the conservative primitive after validating
    ``phi_dot``, so the zero-friction limit is implementation-identical rather
    than merely numerically close.
    """

    value = _finite_gamma(gamma)
    velocity = np.asarray(phi_dot)
    if velocity.size == 0:
        raise ValueError("phi_dot must not be empty")
    if not np.issubdtype(velocity.dtype, np.number) or np.issubdtype(
        velocity.dtype, np.bool_
    ):
        raise TypeError("phi_dot must contain real or complex numeric values")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("phi_dot must contain only finite values")

    conservative = klein_gordon_acceleration(
        phi,
        grid,
        potential,
        boundary=boundary,
    )
    if velocity.shape != conservative.shape:
        raise ValueError("phi_dot must have the same shape as phi")
    if value == 0.0:
        return conservative
    return conservative - value * velocity
