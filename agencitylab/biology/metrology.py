"""Explicit metrological reconstruction from instrument observations to ``u``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from agencitylab.models import ParameterProvenance, ParameterSource

from .models import BiologicalCoordinate, BiologicalMeasurement, BiologicalObservable


def _default_map_provenance() -> ParameterProvenance:
    return ParameterProvenance(
        source=ParameterSource.USER_SUPPLIED,
        note="Explicit calibrated affine measurement map",
    )


@dataclass(frozen=True, slots=True)
class BiologicalMeasurementMap:
    """Versioned calibrated affine map from instrument channels to ``u_hat``.

    This is a generic metrological helper, not a claim that all biological
    measurement maps are affine. More specialized maps can reconstruct a
    :class:`BiologicalObservable` directly while preserving the same explicit
    provenance boundary.
    """

    identifier: str
    version: str
    input_channels: tuple[str, ...]
    input_units: tuple[str, ...]
    output_coordinates: tuple[BiologicalCoordinate, ...]
    matrix: np.ndarray
    offset: np.ndarray | None = None
    provenance: ParameterProvenance = field(default_factory=_default_map_provenance)

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        version = str(self.version).strip()
        if not identifier or not version:
            raise ValueError("measurement map identifier and version must be non-empty")
        if not self.input_channels or not self.output_coordinates:
            raise ValueError("measurement map requires input and output coordinates")
        channels = tuple(str(item).strip() for item in self.input_channels)
        if any(not item for item in channels) or len(set(channels)) != len(channels):
            raise ValueError("measurement map input channels must be unique and non-empty")
        units = tuple(str(item).strip() for item in self.input_units)
        if len(units) != len(channels) or any(not item for item in units):
            raise ValueError("measurement map input_units must match input_channels")
        matrix = np.asarray(self.matrix, dtype=float)
        expected = (len(self.output_coordinates), len(channels))
        if matrix.shape != expected:
            raise ValueError(f"measurement map matrix must have shape {expected}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("measurement map matrix must contain only finite values")
        matrix = np.array(matrix, dtype=float, copy=True)
        matrix.setflags(write=False)
        if self.offset is None:
            offset = np.zeros(len(self.output_coordinates), dtype=float)
        else:
            offset = np.asarray(self.offset, dtype=float)
            if offset.shape != (len(self.output_coordinates),):
                raise ValueError("measurement map offset must have one value per output")
            if not np.all(np.isfinite(offset)):
                raise ValueError("measurement map offset must contain only finite values")
            offset = np.array(offset, dtype=float, copy=True)
        offset.setflags(write=False)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "input_channels", channels)
        object.__setattr__(self, "input_units", units)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)


def reconstruct_observable(
    measurement: BiologicalMeasurement,
    measurement_map: BiologicalMeasurementMap,
) -> BiologicalObservable:
    """Reconstruct a biological observable using an explicit calibrated map."""
    if measurement.channels != measurement_map.input_channels:
        raise ValueError("measurement channels do not match the calibrated measurement map")
    if measurement.units != measurement_map.input_units:
        raise ValueError("measurement units do not match the calibrated measurement map")
    offset = measurement_map.offset
    if offset is None:  # normalized to zeros by BiologicalMeasurementMap.__post_init__
        raise RuntimeError("measurement map offset normalization failed")
    values = measurement.values @ measurement_map.matrix.T + offset

    covariance = None
    if measurement.covariance is not None:
        if measurement.covariance.ndim == 2:
            covariance = (
                measurement_map.matrix
                @ measurement.covariance
                @ measurement_map.matrix.T
            )
        else:
            n_outputs = len(measurement_map.output_coordinates)
            covariance = np.empty(
                (measurement.values.shape[0], n_outputs, n_outputs),
                dtype=float,
            )
            for index, input_covariance in enumerate(measurement.covariance):
                covariance[index] = (
                    measurement_map.matrix
                    @ input_covariance
                    @ measurement_map.matrix.T
                )

    return BiologicalObservable(
        xi=measurement.xi,
        values=values,
        coordinates=measurement_map.output_coordinates,
        coordinate_unit=measurement.coordinate_unit,
        covariance=covariance,
        measurement_map_id=measurement_map.identifier,
        measurement_map_version=measurement_map.version,
    )
