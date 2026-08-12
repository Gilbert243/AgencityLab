"""Conservative classical dynamics for the autonomous Agencity field.

The equation implemented here is the Volume-2 nonlinear Klein-Gordon field
in dimensionless/natural units with ``c = 1``.  This module deliberately
reuses the shared numerical Laplacian and the existing quartic potential;
it does not duplicate either formula.

Scientific status: research. No empirical validation is claimed.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from ..numerics import UniformRectilinearGrid, laplacian
from ..numerics.boundaries import Boundary
from ..physics import QuarticAgencityPotential

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def klein_gordon_acceleration(
    phi: np.ndarray,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Return the conservative Klein-Gordon acceleration of ``phi``.

    In the natural/dimensionless-unit convention used by this research layer,
    the Volume-2 equation is evaluated as the numerical spatial Laplacian minus
    ``potential.gradient(phi)``.  Boundary semantics belong to the existing
    numerical operator and no epsilon or alternative potential is introduced.

    Parameters
    ----------
    phi:
        Finite real or complex spatial field with shape ``grid.shape``.
    grid:
        Existing uniform rectilinear numerical grid.
    potential:
        Existing :class:`QuarticAgencityPotential` supplying the local source
        term through ``gradient``.
    boundary:
        Existing periodic, Dirichlet, or Neumann numerical boundary contract.
    """

    if not isinstance(grid, UniformRectilinearGrid):
        raise TypeError("grid must be a UniformRectilinearGrid")
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")

    array = np.asarray(phi)
    return laplacian(array, grid, boundary=boundary) - potential.gradient(array)
