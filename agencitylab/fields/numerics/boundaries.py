"""Boundary-condition primitives for generic uniform-grid numerics.

The classes here are deliberately small value objects.  They contain no field
physics and are interpreted by the low-level finite-difference operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicBoundary:
    """Periodic boundary condition on every spatial axis."""


@dataclass(frozen=True)
class DirichletBoundary:
    """Scalar Dirichlet value imposed on every spatial boundary face."""

    value: complex | float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("Dirichlet boundary value must be finite")


@dataclass(frozen=True)
class NeumannBoundary:
    """Scalar coordinate derivative imposed at both faces of every axis.

    ``gradient`` denotes the derivative in the positive coordinate direction,
    not the signed outward-normal flux.  This convention lets a linear profile
    ``phi = g*x`` satisfy ``NeumannBoundary(gradient=g)`` at both endpoints.
    """

    gradient: complex | float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.gradient):
            raise ValueError("Neumann boundary gradient must be finite")


Boundary = PeriodicBoundary | DirichletBoundary | NeumannBoundary


def resolve_boundary(boundary: Boundary | str | None) -> Boundary:
    """Return a validated boundary object.

    String aliases are accepted for convenience with homogeneous scalar data:
    ``"periodic"``, ``"dirichlet"``, and ``"neumann"``.
    """

    if boundary is None:
        return PeriodicBoundary()
    if isinstance(boundary, (PeriodicBoundary, DirichletBoundary, NeumannBoundary)):
        return boundary
    if isinstance(boundary, str):
        normalized = boundary.strip().lower()
        if normalized == "periodic":
            return PeriodicBoundary()
        if normalized == "dirichlet":
            return DirichletBoundary()
        if normalized == "neumann":
            return NeumannBoundary()
    raise ValueError("boundary must be periodic, dirichlet, or neumann")
