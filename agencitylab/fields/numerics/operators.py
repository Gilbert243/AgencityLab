"""Second-order spatial operators on uniform rectilinear grids.

These routines operate on one spatial snapshot at a time.  Every array axis is
spatial; no temporal axis is interpreted here.  Real and complex NumPy arrays
are supported without discarding complex components.
"""

from __future__ import annotations

import numpy as np

from .boundaries import (
    Boundary,
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
    resolve_boundary,
)
from .grid import UniformRectilinearGrid


def _validate_field(field: np.ndarray, grid: UniformRectilinearGrid) -> np.ndarray:
    array = np.asarray(field)
    if array.size == 0:
        raise ValueError("field must not be empty")
    if array.shape != grid.shape:
        raise ValueError(f"field shape {array.shape} does not match grid shape {grid.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError("field must contain real or complex numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError("field must contain only finite values")
    return array


def _require_stencil_size(grid: UniformRectilinearGrid, boundary: Boundary) -> None:
    minimum = 4 if isinstance(boundary, DirichletBoundary) else 3
    for axis, count in enumerate(grid.shape):
        if count < minimum:
            raise ValueError(
                f"axis {axis} has {count} points; {type(boundary).__name__} "
                f"requires at least {minimum} points for the implemented stencil"
            )


def _dirichlet_project(field: np.ndarray, value: complex | float) -> np.ndarray:
    dtype = np.result_type(field.dtype, np.asarray(value).dtype)
    projected = np.array(field, dtype=dtype, copy=True)
    for axis in range(projected.ndim):
        lower = [slice(None)] * projected.ndim
        upper = [slice(None)] * projected.ndim
        lower[axis] = 0
        upper[axis] = -1
        projected[tuple(lower)] = value
        projected[tuple(upper)] = value
    return projected


def gradient(
    field: np.ndarray,
    grid: UniformRectilinearGrid,
    boundary: Boundary | str | None = None,
) -> tuple[np.ndarray, ...]:
    """Compute the spatial gradient with second-order finite differences.

    Returns a tuple of ``grid.ndim`` arrays, each with shape ``grid.shape``.
    Interior points use centered differences.  Periodic boundaries use wrapped
    centered differences; Dirichlet boundaries first impose the scalar boundary
    value and then use second-order one-sided edge formulas; Neumann boundaries
    return the prescribed positive-coordinate derivative at both edge faces.
    """

    array = _validate_field(field, grid)
    resolved = resolve_boundary(boundary)
    _require_stencil_size(grid, resolved)

    if isinstance(resolved, DirichletBoundary):
        work = _dirichlet_project(array, resolved.value)
    else:
        work = array

    components: list[np.ndarray] = []
    for axis, spacing in enumerate(grid.spacings):
        if isinstance(resolved, PeriodicBoundary):
            derivative = (
                np.roll(work, -1, axis=axis) - np.roll(work, 1, axis=axis)
            ) / (2.0 * spacing)
        else:
            dtype = np.result_type(work.dtype, float)
            if isinstance(resolved, NeumannBoundary):
                dtype = np.result_type(dtype, np.asarray(resolved.gradient).dtype)
            derivative = np.empty(grid.shape, dtype=dtype)

            center = [slice(None)] * grid.ndim
            plus = [slice(None)] * grid.ndim
            minus = [slice(None)] * grid.ndim
            center[axis] = slice(1, -1)
            plus[axis] = slice(2, None)
            minus[axis] = slice(None, -2)
            derivative[tuple(center)] = (
                work[tuple(plus)] - work[tuple(minus)]
            ) / (2.0 * spacing)

            lower = [slice(None)] * grid.ndim
            upper = [slice(None)] * grid.ndim
            lower[axis] = 0
            upper[axis] = -1
            if isinstance(resolved, NeumannBoundary):
                derivative[tuple(lower)] = resolved.gradient
                derivative[tuple(upper)] = resolved.gradient
            else:
                i0 = [slice(None)] * grid.ndim
                i1 = [slice(None)] * grid.ndim
                i2 = [slice(None)] * grid.ndim
                im0 = [slice(None)] * grid.ndim
                im1 = [slice(None)] * grid.ndim
                im2 = [slice(None)] * grid.ndim
                i0[axis], i1[axis], i2[axis] = 0, 1, 2
                im0[axis], im1[axis], im2[axis] = -1, -2, -3
                derivative[tuple(lower)] = (
                    -3.0 * work[tuple(i0)]
                    + 4.0 * work[tuple(i1)]
                    - work[tuple(i2)]
                ) / (2.0 * spacing)
                derivative[tuple(upper)] = (
                    3.0 * work[tuple(im0)]
                    - 4.0 * work[tuple(im1)]
                    + work[tuple(im2)]
                ) / (2.0 * spacing)
        components.append(derivative)
    return tuple(components)


def laplacian(
    field: np.ndarray,
    grid: UniformRectilinearGrid,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Compute the N-D Laplacian with second-order uniform-grid stencils.

    Periodic axes use wrapped centered second differences.  Dirichlet data are
    imposed on the field boundary before applying centered interior and
    second-order one-sided boundary formulas.  Neumann edges use ghost-point
    elimination consistent with the prescribed positive-coordinate derivative.
    No epsilon is inserted into the stencil.
    """

    array = _validate_field(field, grid)
    resolved = resolve_boundary(boundary)
    _require_stencil_size(grid, resolved)
    work = _dirichlet_project(array, resolved.value) if isinstance(
        resolved, DirichletBoundary
    ) else array

    dtype = np.result_type(work.dtype, float)
    if isinstance(resolved, NeumannBoundary):
        dtype = np.result_type(dtype, np.asarray(resolved.gradient).dtype)
    result = np.zeros(grid.shape, dtype=dtype)

    for axis, spacing in enumerate(grid.spacings):
        h2 = spacing * spacing
        if isinstance(resolved, PeriodicBoundary):
            result += (
                np.roll(work, -1, axis=axis)
                - 2.0 * work
                + np.roll(work, 1, axis=axis)
            ) / h2
            continue

        contribution = np.empty(grid.shape, dtype=dtype)
        center = [slice(None)] * grid.ndim
        plus = [slice(None)] * grid.ndim
        minus = [slice(None)] * grid.ndim
        center[axis] = slice(1, -1)
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        contribution[tuple(center)] = (
            work[tuple(plus)] - 2.0 * work[tuple(center)] + work[tuple(minus)]
        ) / h2

        lower = [slice(None)] * grid.ndim
        upper = [slice(None)] * grid.ndim
        lower[axis] = 0
        upper[axis] = -1
        if isinstance(resolved, NeumannBoundary):
            i0 = [slice(None)] * grid.ndim
            i1 = [slice(None)] * grid.ndim
            im0 = [slice(None)] * grid.ndim
            im1 = [slice(None)] * grid.ndim
            i0[axis], i1[axis] = 0, 1
            im0[axis], im1[axis] = -1, -2
            contribution[tuple(lower)] = (
                2.0 * (work[tuple(i1)] - work[tuple(i0)]) / h2
                - 2.0 * resolved.gradient / spacing
            )
            contribution[tuple(upper)] = (
                2.0 * (work[tuple(im1)] - work[tuple(im0)]) / h2
                + 2.0 * resolved.gradient / spacing
            )
        else:
            i0 = [slice(None)] * grid.ndim
            i1 = [slice(None)] * grid.ndim
            i2 = [slice(None)] * grid.ndim
            i3 = [slice(None)] * grid.ndim
            im0 = [slice(None)] * grid.ndim
            im1 = [slice(None)] * grid.ndim
            im2 = [slice(None)] * grid.ndim
            im3 = [slice(None)] * grid.ndim
            i0[axis], i1[axis], i2[axis], i3[axis] = 0, 1, 2, 3
            im0[axis], im1[axis], im2[axis], im3[axis] = -1, -2, -3, -4
            contribution[tuple(lower)] = (
                2.0 * work[tuple(i0)]
                - 5.0 * work[tuple(i1)]
                + 4.0 * work[tuple(i2)]
                - work[tuple(i3)]
            ) / h2
            contribution[tuple(upper)] = (
                2.0 * work[tuple(im0)]
                - 5.0 * work[tuple(im1)]
                + 4.0 * work[tuple(im2)]
                - work[tuple(im3)]
            ) / h2
        result += contribution
    return result


def gradient_norm_squared(
    field: np.ndarray,
    grid: UniformRectilinearGrid,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Return ``sum_i |partial_i field|**2`` as a real non-negative array."""

    components = gradient(field, grid, boundary=boundary)
    result = np.zeros(grid.shape, dtype=float)
    for component in components:
        result += np.abs(component) ** 2
    return result


def integrate_spatial(density: np.ndarray, grid: UniformRectilinearGrid):
    """Integrate a discrete density by uniform rectangular quadrature.

    The approximation is ``sum(density) * grid.cell_volume``.  This is a
    deliberately simple discrete rectangular rule, not a general high-order
    quadrature method.  Complex densities produce complex integrals.
    """

    array = _validate_field(density, grid)
    return np.sum(array) * grid.cell_volume
