"""Direct complex-field form of the Chapter-15 effective beta evolution.

Volume 2 postulates

    partial_t beta + v . grad(beta)
        = D_diff laplacian(beta) + a beta - b |beta|^2 beta.   (15.2--15.3)

The names in this API deliberately avoid collisions with the canonical dynamic
intensity ``D`` and canonical Agencity flux ``b``.

The amplitude/phase equations (15.4--15.5) are not independently redefined in
this module.  The direct complex equation is the source evolution law.  A
source-normalisation issue in the printed phase equation is documented rather
than silently repaired in code.

Scientific status: research.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from ..numerics import UniformRectilinearGrid, gradient, laplacian
from ..numerics.boundaries import Boundary
from .reaction import _finite_real_scalar, effective_beta_reaction

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _validate_field(beta, grid: UniformRectilinearGrid) -> np.ndarray:
    if not isinstance(grid, UniformRectilinearGrid):
        raise TypeError("grid must be a UniformRectilinearGrid")
    field = np.asarray(beta)
    if field.shape != grid.shape:
        raise ValueError(f"beta shape {field.shape} does not match grid shape {grid.shape}")
    if not np.issubdtype(field.dtype, np.number) or np.issubdtype(field.dtype, np.bool_):
        raise TypeError("beta must contain real or complex numeric values")
    if not np.all(np.isfinite(field)):
        raise ValueError("beta must contain only finite values")
    return field


def _velocity_components(
    velocity,
    grid: UniformRectilinearGrid,
) -> tuple[np.ndarray | float, ...]:
    if velocity is None:
        return tuple(0.0 for _ in range(grid.ndim))

    if isinstance(velocity, np.ndarray) and velocity.shape == (grid.ndim, *grid.shape):
        components = tuple(velocity[index] for index in range(grid.ndim))
    else:
        if not isinstance(velocity, Sequence) or isinstance(velocity, (str, bytes)):
            raise ValueError(
                "velocity must be None or a sequence with one component per spatial axis"
            )
        if len(velocity) != grid.ndim:
            raise ValueError("velocity must contain exactly grid.ndim components")
        components = tuple(velocity)

    validated: list[np.ndarray | float] = []
    for axis, component in enumerate(components):
        array = np.asarray(component)
        if array.ndim == 0:
            if np.iscomplexobj(array):
                raise ValueError("velocity components must be real")
            value = float(array)
            if not np.isfinite(value):
                raise ValueError("velocity components must be finite")
            validated.append(value)
            continue
        if array.shape != grid.shape:
            raise ValueError(
                f"velocity component {axis} shape {array.shape} does not match grid shape {grid.shape}"
            )
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("velocity components must be real numeric values")
        if not np.all(np.isfinite(array)):
            raise ValueError("velocity components must be finite")
        validated.append(np.asarray(array, dtype=float))
    return tuple(validated)


def effective_beta_rhs(
    beta,
    grid: UniformRectilinearGrid,
    *,
    diffusion_coefficient: float,
    linear_coefficient: float,
    saturation_coefficient: float,
    velocity=None,
    boundary: Boundary | str | None = None,
) -> np.ndarray:
    """Return ``partial_t beta`` from Volume-2 Eqs. (15.2--15.3).

    ``diffusion_coefficient`` is the positive coefficient called ``D`` in
    Chapter 15; it is not the canonical signal-derived dynamic intensity.
    ``saturation_coefficient`` is the positive coefficient called ``b`` in
    Eq. (15.3); it is not the canonical flux ``b = P_c beta``.

    ``velocity`` may be omitted, supplied as one real scalar or spatial array
    per axis, or supplied as an array of shape ``(grid.ndim, *grid.shape)``.
    Spatial derivatives and boundary semantics are delegated to the existing
    generic Numerics layer.
    """

    field = _validate_field(beta, grid)
    diffusion = _finite_real_scalar(
        diffusion_coefficient,
        name="diffusion_coefficient",
    )
    if diffusion <= 0.0:
        raise ValueError("diffusion_coefficient must be strictly positive")

    components = _velocity_components(velocity, grid)
    grad = gradient(field, grid, boundary=boundary)
    advection = np.zeros(grid.shape, dtype=np.result_type(field.dtype, float))
    for component, derivative in zip(components, grad, strict=True):
        advection = advection + component * derivative

    diffusion_term = diffusion * laplacian(field, grid, boundary=boundary)
    reaction = effective_beta_reaction(
        field,
        linear_coefficient=linear_coefficient,
        saturation_coefficient=saturation_coefficient,
    )
    return diffusion_term + reaction - advection
