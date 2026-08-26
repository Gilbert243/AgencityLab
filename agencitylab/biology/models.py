"""Data contracts for the experimental biological mapping of Agencity.

These models describe biological references, measurements, observables and
protocols. They do not redefine the canonical Theory of Agencity equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from agencitylab.models import ParameterProvenance, ParameterSource


def _clean_text(value: Any, *, name: str, required: bool = True) -> str:
    text = str(value).strip() if value is not None else ""
    if required and not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _positive(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite")
    return out


def _nonnegative_optional(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return out


def _readonly_array(value: ArrayLike, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _readonly_covariance(
    value: ArrayLike,
    *,
    name: str,
    n_components: int,
    n_samples: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        if array.shape != (n_components, n_components):
            raise ValueError(f"{name} must have shape (m,m)")
    elif array.ndim == 3:
        expected = (n_samples, n_components, n_components)
        if n_samples is None or array.shape != expected:
            raise ValueError(f"{name} must have shape (samples,m,m)")
    else:
        raise ValueError(f"{name} must be two- or three-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    scale = max(1.0, float(np.max(np.abs(array))))
    tolerance = np.finfo(float).eps * scale * 256.0
    if np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")
    output = np.array(array, dtype=float, copy=True)
    output.setflags(write=False)
    return output


def _string_pairs(
    value: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted((str(key).strip(), str(item).strip()) for key, item in items))


def _default_provenance() -> ParameterProvenance:
    return ParameterProvenance(
        source=ParameterSource.USER_SUPPLIED,
        note="Explicit biological protocol value",
    )


class ObservableBlock(str, Enum):
    """Five biological observable blocks defined by the biological mapping."""

    Q = "Q"
    GAMMA = "Gamma"
    DELTA = "Delta"
    PSI = "Psi"
    PHI = "Phi"

    def __str__(self) -> str:
        return self.value


class MetrologyLevel(str, Enum):
    """Deployment level of a biological metrology protocol."""

    A = "A"
    B = "B"
    C = "C"

    def __str__(self) -> str:
        return self.value


class DatasetRole(str, Enum):
    """Role of a dataset or sample set in a biological protocol."""

    DISCOVERY = "discovery"
    CLOSURE = "closure"
    CALIBRATION = "calibration"
    TEST = "test"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BiologicalReference:
    """Common biological reference address ``(R, s)``.

    ``support`` represents Omega; ``causal_input`` and ``causal_output`` define
    the causal port ``pi=(x->y)``. External conditions and applicability-domain
    entries are descriptive protocol metadata and are frozen with the reference.
    """

    taxon: str
    support: str
    causal_input: str
    causal_output: str
    external_conditions: tuple[tuple[str, str], ...] | Mapping[str, Any] = ()
    applicability_domain: tuple[tuple[str, str], ...] | Mapping[str, Any] = ()
    identifier: str = ""
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "taxon", _clean_text(self.taxon, name="taxon"))
        object.__setattr__(self, "support", _clean_text(self.support, name="support"))
        object.__setattr__(
            self, "causal_input", _clean_text(self.causal_input, name="causal_input")
        )
        object.__setattr__(
            self, "causal_output", _clean_text(self.causal_output, name="causal_output")
        )
        object.__setattr__(
            self,
            "external_conditions",
            _string_pairs(self.external_conditions),
        )
        object.__setattr__(
            self,
            "applicability_domain",
            _string_pairs(self.applicability_domain),
        )
        object.__setattr__(
            self,
            "identifier",
            _clean_text(self.identifier, name="identifier", required=False),
        )
        object.__setattr__(self, "version", _clean_text(self.version, name="version"))

    @property
    def causal_port(self) -> tuple[str, str]:
        return self.causal_input, self.causal_output

    @property
    def address(self) -> str:
        reference_id = self.identifier or self.support
        return f"{reference_id}@{self.version}:{self.taxon}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxon": self.taxon,
            "support": self.support,
            "causal_input": self.causal_input,
            "causal_output": self.causal_output,
            "external_conditions": dict(self.external_conditions),
            "applicability_domain": dict(self.applicability_domain),
            "identifier": self.identifier,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class StructuralEpoch:
    """Interval over which relevant biological structure is treated as stable."""

    subject_id: str
    start: float
    end: float
    resolution: str = ""
    criteria: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subject_id", _clean_text(self.subject_id, name="subject_id")
        )
        start = float(self.start)
        end = float(self.end)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("structural epoch requires finite start < end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(
            self,
            "resolution",
            _clean_text(self.resolution, name="resolution", required=False),
        )
        object.__setattr__(self, "criteria", tuple(str(item).strip() for item in self.criteria))
        object.__setattr__(self, "evidence", tuple(str(item).strip() for item in self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "start": self.start,
            "end": self.end,
            "resolution": self.resolution,
            "criteria": list(self.criteria),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class BiologicalCapacity:
    """Individual sustainable characteristic energetic capacity ``P_c``."""

    subject_id: str
    value: float
    structural_epoch: StructuralEpoch
    unit: str = "W"
    challenge_protocol: str = ""
    measurement_method: str = ""
    reference_conditions: tuple[tuple[str, str], ...] | Mapping[str, Any] = ()
    uncertainty: float | None = None
    repeatability: float | None = None
    provenance: ParameterProvenance = field(default_factory=_default_provenance)

    def __post_init__(self) -> None:
        subject_id = _clean_text(self.subject_id, name="subject_id")
        if self.structural_epoch.subject_id != subject_id:
            raise ValueError("capacity subject_id must match structural_epoch subject_id")
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "value", _positive(self.value, name="P_c"))
        object.__setattr__(self, "unit", _clean_text(self.unit, name="unit"))
        object.__setattr__(
            self,
            "challenge_protocol",
            _clean_text(self.challenge_protocol, name="challenge_protocol", required=False),
        )
        object.__setattr__(
            self,
            "measurement_method",
            _clean_text(self.measurement_method, name="measurement_method", required=False),
        )
        object.__setattr__(
            self,
            "reference_conditions",
            _string_pairs(self.reference_conditions),
        )
        object.__setattr__(
            self,
            "uncertainty",
            _nonnegative_optional(self.uncertainty, name="uncertainty"),
        )
        object.__setattr__(
            self,
            "repeatability",
            _nonnegative_optional(self.repeatability, name="repeatability"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "value": self.value,
            "unit": self.unit,
            "structural_epoch": self.structural_epoch.to_dict(),
            "challenge_protocol": self.challenge_protocol,
            "measurement_method": self.measurement_method,
            "reference_conditions": dict(self.reference_conditions),
            "uncertainty": self.uncertainty,
            "repeatability": self.repeatability,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BiologicalTimeReference:
    """Frozen class/reference characteristic time ``tau(R, s)``."""

    reference: BiologicalReference
    q_reference: float
    tau: float
    sample_size: int
    dispersion: float | None = None
    unit: str = "s"
    source_ids: tuple[str, ...] = ()
    provenance: ParameterProvenance = field(default_factory=_default_provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_reference", _positive(self.q_reference, name="q_reference"))
        object.__setattr__(self, "tau", _positive(self.tau, name="tau"))
        object.__setattr__(self, "unit", _clean_text(self.unit, name="time reference unit"))
        source_ids = tuple(_clean_text(item, name="source_id") for item in self.source_ids)
        if source_ids and len(source_ids) != self.sample_size:
            raise ValueError("time reference source_ids must match sample_size")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("time reference source_ids must be unique")
        object.__setattr__(self, "source_ids", source_ids)
        if int(self.sample_size) != self.sample_size or self.sample_size < 1:
            raise ValueError("sample_size must be a positive integer")
        object.__setattr__(self, "sample_size", int(self.sample_size))
        object.__setattr__(
            self,
            "dispersion",
            _nonnegative_optional(self.dispersion, name="dispersion"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "q_reference": self.q_reference,
            "tau": self.tau,
            "sample_size": self.sample_size,
            "dispersion": self.dispersion,
            "unit": self.unit,
            "source_ids": list(self.source_ids),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BiologicalMemoryReference:
    """Frozen class/reference CRM causal depth ``w(R, s)``."""

    reference: BiologicalReference
    w: float
    individual_causal_times: tuple[float, ...]
    dispersion: float | None = None
    unit: str = "s"
    source_ids: tuple[str, ...] = ()
    provenance: ParameterProvenance = field(default_factory=_default_provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "w", _positive(self.w, name="w"))
        times = tuple(
            _positive(item, name="individual causal time")
            for item in self.individual_causal_times
        )
        object.__setattr__(self, "unit", _clean_text(self.unit, name="memory reference unit"))
        if not times:
            raise ValueError("individual_causal_times must be non-empty")
        source_ids = tuple(_clean_text(item, name="source_id") for item in self.source_ids)
        if source_ids and len(source_ids) != len(times):
            raise ValueError("memory reference source_ids must match causal-time count")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("memory reference source_ids must be unique")
        object.__setattr__(self, "individual_causal_times", times)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(
            self,
            "dispersion",
            _nonnegative_optional(self.dispersion, name="dispersion"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "w": self.w,
            "individual_causal_times": list(self.individual_causal_times),
            "dispersion": self.dispersion,
            "unit": self.unit,
            "source_ids": list(self.source_ids),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BiologicalResolution:
    """Declared biological measurement resolution ``R_org``."""

    spatial: float | None = None
    material: float | None = None
    temporal: float | None = None
    spatial_unit: str = ""
    material_unit: str = ""
    temporal_unit: str = "s"

    def __post_init__(self) -> None:
        for name in ("spatial", "material", "temporal"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name=name))
        for name in ("spatial_unit", "material_unit", "temporal_unit"):
            object.__setattr__(
                self,
                name,
                _clean_text(getattr(self, name), name=name, required=False),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spatial": self.spatial,
            "material": self.material,
            "temporal": self.temporal,
            "spatial_unit": self.spatial_unit,
            "material_unit": self.material_unit,
            "temporal_unit": self.temporal_unit,
        }


@dataclass(frozen=True, slots=True)
class BiologicalCoordinate:
    """One reconstructed coordinate of the biological observable ``u``."""

    name: str
    block: ObservableBlock | str
    unit: str
    A_ref: float | None = None
    description: str = ""
    provenance: ParameterProvenance = field(default_factory=_default_provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, name="coordinate name"))
        block = (
            self.block
            if isinstance(self.block, ObservableBlock)
            else ObservableBlock(self.block)
        )
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "unit", _clean_text(self.unit, name="coordinate unit"))
        if self.A_ref is not None:
            object.__setattr__(self, "A_ref", _positive(self.A_ref, name=f"A_ref[{self.name}]"))
        object.__setattr__(
            self,
            "description",
            _clean_text(self.description, name="description", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "block": (
                self.block.value
                if isinstance(self.block, ObservableBlock)
                else ObservableBlock(self.block).value
            ),
            "unit": self.unit,
            "A_ref": self.A_ref,
            "description": self.description,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BiologicalMeasurement:
    """Instrument observations ``Z`` before metrological reconstruction."""

    xi: np.ndarray
    values: np.ndarray
    channels: tuple[str, ...]
    units: tuple[str, ...]
    instrument_id: str
    coordinate_unit: str = "s"
    uncertainty: np.ndarray | None = None
    covariance: np.ndarray | None = None
    provenance: ParameterProvenance = field(default_factory=_default_provenance)

    def __post_init__(self) -> None:
        xi = _readonly_array(self.xi, name="xi", ndim=1)
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        values = _readonly_array(values, name="values", ndim=2)
        if values.shape[0] != xi.size:
            raise ValueError("measurement values sample dimension must match xi")
        if len(self.channels) != values.shape[1] or len(self.units) != values.shape[1]:
            raise ValueError("channels and units must have one entry per measurement column")
        channels = tuple(_clean_text(item, name="channel") for item in self.channels)
        if len(set(channels)) != len(channels):
            raise ValueError("measurement channel names must be unique")
        units = tuple(_clean_text(item, name="unit") for item in self.units)
        instrument_id = _clean_text(self.instrument_id, name="instrument_id")
        coordinate_unit = _clean_text(self.coordinate_unit, name="coordinate_unit")
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = np.asarray(self.uncertainty, dtype=float)
            if uncertainty.ndim == 1:
                uncertainty = uncertainty[:, None]
            uncertainty = _readonly_array(uncertainty, name="uncertainty", ndim=2)
            if uncertainty.shape != values.shape:
                raise ValueError("measurement uncertainty must match values shape")
            if np.any(uncertainty < 0.0):
                raise ValueError("measurement uncertainty must be non-negative")
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", units)
        covariance = None
        if self.covariance is not None:
            covariance = _readonly_covariance(
                self.covariance,
                name="measurement covariance",
                n_components=values.shape[1],
                n_samples=xi.size,
            )
        object.__setattr__(self, "instrument_id", instrument_id)
        object.__setattr__(self, "coordinate_unit", coordinate_unit)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, slots=True)
class BiologicalObservable:
    """Reconstructed biological observable ``u_hat`` with explicit coordinates."""

    xi: np.ndarray
    values: np.ndarray
    coordinates: tuple[BiologicalCoordinate, ...]
    coordinate_unit: str = "s"
    covariance: np.ndarray | None = None
    resolution: BiologicalResolution | None = None
    measurement_map_id: str = ""
    measurement_map_version: str = ""

    def __post_init__(self) -> None:
        xi = _readonly_array(self.xi, name="xi", ndim=1)
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        values = _readonly_array(values, name="values", ndim=2)
        if values.shape[0] != xi.size:
            raise ValueError("observable values sample dimension must match xi")
        if len(self.coordinates) != values.shape[1]:
            raise ValueError("coordinates must contain one entry per observable column")
        names = [coordinate.name for coordinate in self.coordinates]
        if len(set(names)) != len(names):
            raise ValueError("observable coordinate names must be unique")
        covariance = None
        if self.covariance is not None:
            covariance = _readonly_covariance(
                self.covariance,
                name="observable covariance",
                n_components=values.shape[1],
                n_samples=xi.size,
            )
        coordinate_unit = _clean_text(self.coordinate_unit, name="coordinate_unit")
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinate_unit", coordinate_unit)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self,
            "measurement_map_id",
            _clean_text(self.measurement_map_id, name="measurement_map_id", required=False),
        )
        object.__setattr__(
            self,
            "measurement_map_version",
            _clean_text(
                self.measurement_map_version,
                name="measurement_map_version",
                required=False,
            ),
        )

    @property
    def n_components(self) -> int:
        return self.values.shape[1]

    @property
    def reference_amplitudes(self) -> tuple[float | None, ...]:
        return tuple(coordinate.A_ref for coordinate in self.coordinates)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Explicit discovery/closure/calibration/test sample identifiers."""

    discovery: frozenset[str] = frozenset()
    closure: frozenset[str] = frozenset()
    calibration: frozenset[str] = frozenset()
    test: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("discovery", "closure", "calibration", "test"):
            values = frozenset(str(item).strip() for item in getattr(self, name))
            if "" in values:
                raise ValueError(f"{name} sample identifiers must be non-empty")
            object.__setattr__(self, name, values)

    def overlaps(self) -> dict[str, frozenset[str]]:
        roles = {
            "discovery": self.discovery,
            "closure": self.closure,
            "calibration": self.calibration,
            "test": self.test,
        }
        names = tuple(roles)
        overlaps: dict[str, frozenset[str]] = {}
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                values = roles[left] & roles[right]
                if values:
                    overlaps[f"{left}_{right}"] = values
        return overlaps

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "discovery": sorted(self.discovery),
            "closure": sorted(self.closure),
            "calibration": sorted(self.calibration),
            "test": sorted(self.test),
        }


@dataclass(frozen=True, slots=True)
class BiologicalProtocol:
    """Complete biological mapping required before Agencity computation."""

    reference: BiologicalReference
    capacity: BiologicalCapacity
    time_reference: BiologicalTimeReference
    memory_reference: BiologicalMemoryReference
    coordinates: tuple[BiologicalCoordinate, ...]
    dataset_split: DatasetSplit = field(default_factory=DatasetSplit)
    measurement_map_id: str = ""
    measurement_map_version: str = ""
    protocol_id: str = "biology"
    version: str = "1"
    metrology_level: MetrologyLevel | str | None = None
    target_informed_calibration: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.coordinates:
            raise ValueError("biology protocol requires at least one observable coordinate")
        names = [coordinate.name for coordinate in self.coordinates]
        if len(set(names)) != len(names):
            raise ValueError("biology protocol coordinate names must be unique")
        if self.time_reference.reference != self.reference:
            raise ValueError("time reference must use the protocol biological reference")
        if self.memory_reference.reference != self.reference:
            raise ValueError("memory reference must use the protocol biological reference")
        object.__setattr__(
            self,
            "measurement_map_id",
            _clean_text(self.measurement_map_id, name="measurement_map_id", required=False),
        )
        object.__setattr__(
            self,
            "measurement_map_version",
            _clean_text(
                self.measurement_map_version,
                name="measurement_map_version",
                required=False,
            ),
        )
        if bool(self.measurement_map_id) != bool(self.measurement_map_version):
            raise ValueError("measurement map id and version must be supplied together")
        object.__setattr__(self, "protocol_id", _clean_text(self.protocol_id, name="protocol_id"))
        object.__setattr__(self, "version", _clean_text(self.version, name="version"))
        if self.metrology_level is not None:
            level = (
                self.metrology_level
                if isinstance(self.metrology_level, MetrologyLevel)
                else MetrologyLevel(self.metrology_level)
            )
            object.__setattr__(self, "metrology_level", level)
        object.__setattr__(
            self,
            "notes",
            _clean_text(self.notes, name="notes", required=False),
        )

    @property
    def reference_amplitudes(self) -> tuple[float | None, ...]:
        return tuple(coordinate.A_ref for coordinate in self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.to_dict(),
            "capacity": self.capacity.to_dict(),
            "time_reference": self.time_reference.to_dict(),
            "memory_reference": self.memory_reference.to_dict(),
            "coordinates": [coordinate.to_dict() for coordinate in self.coordinates],
            "dataset_split": self.dataset_split.to_dict(),
            "measurement_map_id": self.measurement_map_id,
            "measurement_map_version": self.measurement_map_version,
            "protocol_id": self.protocol_id,
            "version": self.version,
            "metrology_level": (
                self.metrology_level.value
                if isinstance(self.metrology_level, MetrologyLevel)
                else self.metrology_level
            ),
            "target_informed_calibration": self.target_informed_calibration,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FrozenBiologicalProtocol:
    """Immutable, content-addressed biological protocol snapshot."""

    protocol: BiologicalProtocol
    content_hash: str
    frozen_at: str


@dataclass(frozen=True, slots=True)
class ScientificValidation:
    """Validation report for a biological mapping or protocol."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BiologicalAgencityResult:
    """Biological context wrapped around a canonical/multivariate result."""

    subject_id: str
    protocol_hash: str
    mode: str
    result: Any
    P_c_components: tuple[float, ...] | None = None
