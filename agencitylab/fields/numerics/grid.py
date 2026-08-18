"""Uniform rectilinear grids for generic field numerics.

This module is numerical infrastructure only.  It defines no Agencity field
physics and supports uniform rectilinear spatial grids exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, init=False)
class UniformRectilinearGrid:
    """Lightweight N-D uniform rectilinear grid.

    Construct the grid either from explicit one-dimensional ``axes`` or from
    ``shape`` and ``spacings`` with optional ``origins``.  Every explicit axis
    must be finite, strictly increasing, and uniformly spaced.  Non-uniform
    grids are intentionally unsupported in this numerical infrastructure.

    Parameters
    ----------
    axes:
        Sequence of one-dimensional coordinate arrays.
    shape:
        Number of grid points on each spatial axis.  Used only when ``axes``
        is omitted.
    spacings:
        Positive uniform spacing for each spatial axis.  Used only when
        ``axes`` is omitted.
    origins:
        Coordinate of the first point of each generated axis.  Defaults to
        zero on every axis.
    """

    axes: tuple[np.ndarray, ...]
    shape: tuple[int, ...]
    spacings: tuple[float, ...]
    origins: tuple[float, ...]
    extent: tuple[tuple[float, float], ...]
    cell_volume: float

    def __init__(
        self,
        axes: Sequence[Iterable[float]] | None = None,
        *,
        shape: Sequence[int] | None = None,
        spacings: Sequence[float] | None = None,
        origins: Sequence[float] | None = None,
    ) -> None:
        if axes is not None:
            if shape is not None or spacings is not None or origins is not None:
                raise ValueError("axes cannot be combined with shape, spacings, or origins")
            parsed_axes = self._validate_axes(axes)
        else:
            parsed_axes = self._build_axes(shape=shape, spacings=spacings, origins=origins)

        parsed_shape = tuple(int(axis.size) for axis in parsed_axes)
        parsed_spacings = tuple(float(axis[1] - axis[0]) for axis in parsed_axes)
        parsed_origins = tuple(float(axis[0]) for axis in parsed_axes)
        parsed_extent = tuple((float(axis[0]), float(axis[-1])) for axis in parsed_axes)
        cell_volume = float(np.prod(parsed_spacings, dtype=float))

        object.__setattr__(self, "axes", parsed_axes)
        object.__setattr__(self, "shape", parsed_shape)
        object.__setattr__(self, "spacings", parsed_spacings)
        object.__setattr__(self, "origins", parsed_origins)
        object.__setattr__(self, "extent", parsed_extent)
        object.__setattr__(self, "cell_volume", cell_volume)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""

        return len(self.axes)

    @property
    def volume_element(self) -> float:
        """Alias for the uniform discrete volume element."""

        return self.cell_volume

    @staticmethod
    def _validate_axes(axes: Sequence[Iterable[float]]) -> tuple[np.ndarray, ...]:
        if len(axes) == 0:
            raise ValueError("at least one spatial axis is required")
        validated: list[np.ndarray] = []
        for index, values in enumerate(axes):
            axis = np.asarray(values, dtype=float)
            if axis.ndim != 1:
                raise ValueError(f"axis {index} must be one-dimensional")
            if axis.size < 2:
                raise ValueError(f"axis {index} must contain at least two points")
            if not np.all(np.isfinite(axis)):
                raise ValueError(f"axis {index} must contain only finite coordinates")
            differences = np.diff(axis)
            if np.any(differences <= 0.0):
                raise ValueError(f"axis {index} must be strictly increasing")
            spacing = float(differences[0])
            tolerance = float(max(abs(spacing) * 1e-12, np.finfo(float).eps * 32.0))
            if not np.allclose(differences, spacing, rtol=1e-10, atol=tolerance):
                raise ValueError(f"axis {index} must be uniformly spaced")
            copy = np.array(axis, dtype=float, copy=True)
            copy.setflags(write=False)
            validated.append(copy)
        return tuple(validated)

    @classmethod
    def _build_axes(
        cls,
        *,
        shape: Sequence[int] | None,
        spacings: Sequence[float] | None,
        origins: Sequence[float] | None,
    ) -> tuple[np.ndarray, ...]:
        if shape is None or spacings is None:
            raise ValueError("provide either axes or both shape and spacings")
        parsed_shape = tuple(shape)
        parsed_spacings = tuple(spacings)
        if len(parsed_shape) == 0:
            raise ValueError("shape must contain at least one spatial dimension")
        if len(parsed_shape) != len(parsed_spacings):
            raise ValueError("shape and spacings must have the same length")
        if origins is None:
            parsed_origins = (0.0,) * len(parsed_shape)
        else:
            parsed_origins = tuple(origins)
            if len(parsed_origins) != len(parsed_shape):
                raise ValueError("origins must have the same length as shape")

        generated: list[np.ndarray] = []
        for index, (count, spacing, origin) in enumerate(
            zip(parsed_shape, parsed_spacings, parsed_origins)
        ):
            if isinstance(count, (bool, np.bool_)) or int(count) != count or int(count) < 2:
                raise ValueError(f"shape[{index}] must be an integer >= 2")
            spacing_value = float(spacing)
            origin_value = float(origin)
            if not np.isfinite(spacing_value) or spacing_value <= 0.0:
                raise ValueError(f"spacings[{index}] must be finite and positive")
            if not np.isfinite(origin_value):
                raise ValueError(f"origins[{index}] must be finite")
            axis = origin_value + spacing_value * np.arange(int(count), dtype=float)
            axis.setflags(write=False)
            generated.append(axis)
        return tuple(generated)
