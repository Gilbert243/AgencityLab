"""Lightweight contracts for future dynamical Agencity field extensions.

This module defines data models only. It does not implement a beta-to-phi bridge,
physical potentials, spatial differential operators, PDEs, thermodynamics,
gravity, quantisation, or cosmology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from agencitylab.scientific_status import ScientificStatus


class ParameterSource(str, Enum):
    """Allowed origins for numerical or physical model parameters."""

    USER_SUPPLIED = "user_supplied"
    NAMED_PHYSICAL_CONTEXT = "named_physical_context"
    DIMENSIONLESS_BENCHMARK = "dimensionless_benchmark"
    SOURCE_DOCUMENT_REFERENCE = "source_document_reference"
    DERIVED_MATHEMATICALLY = "derived_mathematically"
    IMPLEMENTATION_CONVENTION = "implementation_convention"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ParameterProvenance:
    """Minimal provenance record for one parameter value or convention."""

    source: ParameterSource | str
    note: str = ""
    reference: str = ""

    def __post_init__(self) -> None:
        try:
            source = (
                self.source
                if isinstance(self.source, ParameterSource)
                else ParameterSource(self.source)
            )
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ParameterSource)
            raise ValueError(
                f"unknown parameter provenance source; expected one of: {allowed}"
            ) from exc
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "note", str(self.note).strip())
        object.__setattr__(self, "reference", str(self.reference).strip())

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source.value,
            "note": self.note,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterProvenance":
        return cls(
            source=data["source"],
            note=data.get("note", ""),
            reference=data.get("reference", ""),
        )


def _contains_callable(value: Any) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_callable(key) or _contains_callable(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_callable(item) for item in value)
    return False


def _clean_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    data = dict(value or {})
    if _contains_callable(data):
        raise ValueError(f"{name} must not contain Python callables")
    return data


def _validate_units(value: str) -> str:
    value = str(value).strip()
    if value not in {"natural_units", "dimensionless"}:
        raise ValueError("units_convention must be 'natural_units' or 'dimensionless'")
    return value


def _coerce_status(value: ScientificStatus | str) -> ScientificStatus:
    return value if isinstance(value, ScientificStatus) else ScientificStatus(value)


def _validate_research_status(value: ScientificStatus | str) -> ScientificStatus:
    status = _coerce_status(value)
    if status is not ScientificStatus.RESEARCH:
        raise ValueError(
            "dynamical Agencity field models must have scientific status 'research'"
        )
    return status


def _validate_axes(
    axes: tuple[np.ndarray, ...] | None,
    spatial_shape: tuple[int, ...],
) -> tuple[np.ndarray, ...] | None:
    if axes is None:
        return None
    if len(axes) != len(spatial_shape):
        raise ValueError("spatial_axes must contain one axis per spatial dimension")
    validated: list[np.ndarray] = []
    for index, (axis, size) in enumerate(zip(axes, spatial_shape)):
        arr = np.asarray(axis, dtype=float)
        if arr.ndim != 1 or arr.size != size:
            raise ValueError(
                f"spatial_axes[{index}] must be one-dimensional with length {size}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"spatial_axes[{index}] must contain only finite values"
            )
        validated.append(arr)
    return tuple(validated)


def _normalize_provenance(
    value: Mapping[str, ParameterProvenance | Mapping[str, Any]] | None,
) -> dict[str, ParameterProvenance]:
    output: dict[str, ParameterProvenance] = {}
    for name, item in dict(value or {}).items():
        key = str(name)
        output[key] = (
            item
            if isinstance(item, ParameterProvenance)
            else ParameterProvenance.from_dict(item)
        )
    return output


@dataclass(slots=True)
class FieldModelMetadata:
    """Serializable scientific metadata shared by all future field layers."""

    model_name: str
    scientific_status: ScientificStatus | str = ScientificStatus.RESEARCH
    theory_source: str = ""
    assumptions: tuple[str, ...] = ()
    units_convention: str = "dimensionless"
    parameter_provenance: dict[str, ParameterProvenance] = field(default_factory=dict)
    software_version: str = ""
    numerical_method: str = ""
    boundary_condition: str = ""
    grid_description: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.model_name = str(self.model_name).strip()
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        self.scientific_status = _coerce_status(self.scientific_status)
        self.units_convention = _validate_units(self.units_convention)
        self.theory_source = str(self.theory_source).strip()
        self.assumptions = tuple(str(item).strip() for item in self.assumptions)
        self.parameter_provenance = _normalize_provenance(self.parameter_provenance)
        for name in (
            "software_version",
            "numerical_method",
            "boundary_condition",
            "grid_description",
            "notes",
        ):
            setattr(self, name, str(getattr(self, name)).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "scientific_status": self.scientific_status.value,
            "theory_source": self.theory_source,
            "assumptions": list(self.assumptions),
            "units_convention": self.units_convention,
            "parameter_provenance": {
                key: item.to_dict()
                for key, item in self.parameter_provenance.items()
            },
            "software_version": self.software_version,
            "numerical_method": self.numerical_method,
            "boundary_condition": self.boundary_condition,
            "grid_description": self.grid_description,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldModelMetadata":
        return cls(**dict(data))


@dataclass(slots=True)
class DynamicalAgencityFieldState:
    """One autonomous ``phi(x,t)`` spatial snapshot.

    The object is a research data contract, not an observed field result.
    """

    phi: np.ndarray
    time: float
    spatial_shape: tuple[int, ...]
    phi_dot: np.ndarray | None = None
    spatial_axes: tuple[np.ndarray, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scientific_status: ScientificStatus | str = ScientificStatus.RESEARCH
    model_name: str = "dynamical_agencity_field_state"
    units_convention: str = "dimensionless"

    def __post_init__(self) -> None:
        raw_phi = np.asarray(self.phi)
        dtype = complex if np.iscomplexobj(raw_phi) else float
        self.phi = np.asarray(raw_phi, dtype=dtype)
        if self.phi.ndim < 1 or not np.all(np.isfinite(self.phi)):
            raise ValueError("phi must be a finite spatial array")

        self.spatial_shape = tuple(int(size) for size in self.spatial_shape)
        if (
            any(size <= 0 for size in self.spatial_shape)
            or self.phi.shape != self.spatial_shape
        ):
            raise ValueError("spatial_shape must match phi.shape")

        if self.phi_dot is not None:
            raw_dot = np.asarray(self.phi_dot)
            dot_dtype = complex if np.iscomplexobj(raw_dot) else float
            self.phi_dot = np.asarray(raw_dot, dtype=dot_dtype)
            if (
                self.phi_dot.shape != self.phi.shape
                or not np.all(np.isfinite(self.phi_dot))
            ):
                raise ValueError("phi_dot must be finite and have the same shape as phi")

        self.time = float(self.time)
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        self.spatial_axes = _validate_axes(self.spatial_axes, self.spatial_shape)
        self.metadata = _clean_mapping(self.metadata, name="metadata")
        self.scientific_status = _validate_research_status(self.scientific_status)
        self.model_name = str(self.model_name).strip()
        if self.model_name != "dynamical_agencity_field_state":
            raise ValueError("unexpected dynamical field state model_name")
        self.units_convention = _validate_units(self.units_convention)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phi": self.phi.copy(),
            "phi_dot": None if self.phi_dot is None else self.phi_dot.copy(),
            "time": self.time,
            "spatial_shape": self.spatial_shape,
            "spatial_axes": (
                None
                if self.spatial_axes is None
                else tuple(axis.copy() for axis in self.spatial_axes)
            ),
            "metadata": dict(self.metadata),
            "scientific_status": self.scientific_status.value,
            "model_name": self.model_name,
            "units_convention": self.units_convention,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DynamicalAgencityFieldState":
        return cls(**dict(data))


@dataclass(slots=True)
class DynamicalAgencityFieldSolution:
    """Numerical trajectory of an autonomous dynamical Agencity field.

    Shape convention: ``phi.shape == (n_time, *spatial_shape)``. The optional
    ``phi_dot`` follows exactly the same convention.
    """

    times: np.ndarray
    phi: np.ndarray
    spatial_shape: tuple[int, ...]
    phi_dot: np.ndarray | None = None
    spatial_axes: tuple[np.ndarray, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_provenance: dict[str, ParameterProvenance] = field(default_factory=dict)
    dynamics_name: str = ""
    boundary_name: str = ""
    scientific_status: ScientificStatus | str = ScientificStatus.RESEARCH
    solver_metadata: dict[str, Any] = field(default_factory=dict)
    units_convention: str = "dimensionless"

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        if self.times.ndim != 1 or self.times.size < 1:
            raise ValueError("times must be a non-empty one-dimensional array")
        if (
            not np.all(np.isfinite(self.times))
            or np.any(np.diff(self.times) <= 0.0)
        ):
            raise ValueError("times must contain finite strictly increasing values")

        raw_phi = np.asarray(self.phi)
        dtype = complex if np.iscomplexobj(raw_phi) else float
        self.phi = np.asarray(raw_phi, dtype=dtype)
        self.spatial_shape = tuple(int(size) for size in self.spatial_shape)
        if any(size <= 0 for size in self.spatial_shape):
            raise ValueError("spatial_shape must contain strictly positive sizes")
        expected = (self.times.size, *self.spatial_shape)
        if self.phi.shape != expected or not np.all(np.isfinite(self.phi)):
            raise ValueError(f"phi must be finite and have shape {expected}")

        if self.phi_dot is not None:
            raw_dot = np.asarray(self.phi_dot)
            dot_dtype = complex if np.iscomplexobj(raw_dot) else float
            self.phi_dot = np.asarray(raw_dot, dtype=dot_dtype)
            if (
                self.phi_dot.shape != expected
                or not np.all(np.isfinite(self.phi_dot))
            ):
                raise ValueError(f"phi_dot must be finite and have shape {expected}")

        self.spatial_axes = _validate_axes(self.spatial_axes, self.spatial_shape)
        self.metadata = _clean_mapping(self.metadata, name="metadata")
        self.parameters = _clean_mapping(self.parameters, name="parameters")
        self.parameter_provenance = _normalize_provenance(
            self.parameter_provenance
        )
        self.solver_metadata = _clean_mapping(
            self.solver_metadata, name="solver_metadata"
        )
        self.dynamics_name = str(self.dynamics_name).strip()
        self.boundary_name = str(self.boundary_name).strip()
        self.scientific_status = _validate_research_status(self.scientific_status)
        self.units_convention = _validate_units(self.units_convention)

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.copy(),
            "phi": self.phi.copy(),
            "phi_dot": None if self.phi_dot is None else self.phi_dot.copy(),
            "spatial_shape": self.spatial_shape,
            "spatial_axes": (
                None
                if self.spatial_axes is None
                else tuple(axis.copy() for axis in self.spatial_axes)
            ),
            "metadata": dict(self.metadata),
            "parameters": dict(self.parameters),
            "parameter_provenance": {
                key: item.to_dict()
                for key, item in self.parameter_provenance.items()
            },
            "dynamics_name": self.dynamics_name,
            "boundary_name": self.boundary_name,
            "scientific_status": self.scientific_status.value,
            "solver_metadata": dict(self.solver_metadata),
            "units_convention": self.units_convention,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DynamicalAgencityFieldSolution":
        return cls(**dict(data))
