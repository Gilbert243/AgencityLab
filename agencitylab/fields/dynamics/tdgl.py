"""Overdamped time-dependent Ginzburg-Landau Agencity dynamics.

The research equation is the overdamped Volume-2 limit of the dissipative
Klein-Gordon model.  It reuses the shared numerical Laplacian and the existing
quartic potential-gradient source without redefining either formula.

Scientific status: research. No empirical validation is claimed.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from ..numerics import UniformRectilinearGrid, laplacian
from ..numerics.boundaries import Boundary
from ..physics import QuarticAgencityPotential
from .dissipative import _finite_gamma

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def tdgl_rhs(
    phi: np.ndarray,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    gamma: float,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Return the overdamped TDGL right-hand side ``dphi/dt``.

    ``gamma`` is a physical/model coefficient supplied by the caller.  This
    overdamped equation requires ``gamma > 0`` exactly; no epsilon is inserted
    into the denominator.
    """

    value = _finite_gamma(gamma)
    if value == 0.0:
        raise ValueError("TDGL requires gamma to be strictly positive")
    if not isinstance(grid, UniformRectilinearGrid):
        raise TypeError("grid must be a UniformRectilinearGrid")
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")

    array = np.asarray(phi)
    return (
        laplacian(array, grid, boundary=boundary) - potential.gradient(array)
    ) / value
