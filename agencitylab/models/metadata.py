"""Scientific metadata model for AgencityLab.

Metadata carries physical/contextual information required to reproduce a
computation. Unit fields are labels only: AgencityLab does not silently convert
magnitudes between unit systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


def _clean_text(value: Any, *, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _positive_optional(value: Any, *, name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return out


def _agencity_flux_unit(power_unit: str) -> str:
    """Return the theory-facing flux label corresponding to a power unit."""
    return f"{power_unit}·nat" if power_unit else ""


@dataclass(slots=True)
class ExperimentMetadata:
    """Reproducibility metadata for one Agencity computation.

    ``unit`` labels the observable and therefore also ``A_ref``.
    ``coordinate_unit`` labels ``xi`` and therefore also ``tau``.
    ``power_unit`` labels ``P_c``. The observable agencity flux ``b`` carries the
    corresponding informational-power label ``power_unit · nat`` (for example,
    ``W·nat``). These labels are descriptive contracts; no automatic unit
    conversion is performed.

    ``characteristic_power`` stores a scalar physical ``P_c`` when one exists.
    A time-varying externally supplied ``P_c(t)`` lives on ``AgencityResult.P_c``
    and is identified through metadata ``extra`` rather than being collapsed to a
    scalar.

    ``activity_factor`` and ``resolution_scale`` remain serializable for legacy
    compatibility and observational metadata, but they do not modify the
    canonical v0.3 computation path.
    """

    title: str = ""
    description: str = ""
    author: str = ""
    domain: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)

    coordinate_name: str = "t"
    coordinate_unit: str = ""
    signal_name: str = "u"
    observable_kind: str = ""
    unit: str = ""
    power_unit: str = ""

    reference_amplitude: Optional[float] = None
    characteristic_time: Optional[float] = None
    characteristic_power: Optional[float] = None
    memory_window: Optional[float] = None

    # Legacy/observational metadata. These do not alter canonical equations.
    activity_factor: Optional[float] = None
    resolution_scale: Optional[float] = None

    system_type: str = ""
    mechanism: str = ""
    environment: str = ""
    geometry: str = ""

    component_units: list[str] = field(default_factory=list)
    component_kinds: list[str] = field(default_factory=list)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "title",
            "description",
            "author",
            "domain",
            "source",
            "coordinate_name",
            "coordinate_unit",
            "signal_name",
            "observable_kind",
            "unit",
            "power_unit",
            "system_type",
            "mechanism",
            "environment",
            "geometry",
            "created_at",
        ):
            setattr(self, name, _clean_text(getattr(self, name), name=name))

        self.tags = [str(value).strip() for value in self.tags]
        self.component_units = [str(value).strip() for value in self.component_units]
        self.component_kinds = [str(value).strip() for value in self.component_kinds]
        self.extra = dict(self.extra or {})

        self.reference_amplitude = _positive_optional(
            self.reference_amplitude, name="reference_amplitude"
        )
        self.characteristic_time = _positive_optional(
            self.characteristic_time, name="characteristic_time"
        )
        self.characteristic_power = _positive_optional(
            self.characteristic_power, name="characteristic_power"
        )
        self.memory_window = _positive_optional(self.memory_window, name="memory_window")
        self.activity_factor = _positive_optional(
            self.activity_factor, name="activity_factor"
        )
        self.resolution_scale = _positive_optional(
            self.resolution_scale, name="resolution_scale"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialization-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "ExperimentMetadata" | None) -> "ExperimentMetadata":
        """Create metadata while preserving unknown fields in ``extra``."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return cls(**data.to_dict())
        if not isinstance(data, dict):
            raise ValueError("metadata must be a dictionary, ExperimentMetadata, or None")

        payload = dict(data)
        allowed = {f.name for f in fields(cls)}
        known: Dict[str, Any] = {}
        for key in list(payload):
            if key in allowed:
                known[key] = payload.pop(key)

        extra = dict(known.get("extra", {}) or {})
        extra.update(payload)
        known["extra"] = extra
        return cls(**known)

    def with_updates(self, **updates: Any) -> "ExperimentMetadata":
        """Return a validated copy with selected fields updated."""
        payload = self.to_dict()
        payload.update(updates)
        return ExperimentMetadata.from_dict(payload)

    def has_reference_amplitude(self) -> bool:
        return self.reference_amplitude is not None

    def has_characteristic_time(self) -> bool:
        return self.characteristic_time is not None

    def has_characteristic_power(self) -> bool:
        return self.characteristic_power is not None

    def has_activity_factor(self) -> bool:
        return self.activity_factor is not None

    def has_resolution_scale(self) -> bool:
        return self.resolution_scale is not None

    @property
    def agencity_unit(self) -> str:
        """Unit label for the observable flux ``b``."""
        return _agencity_flux_unit(self.power_unit)

    def reference_context(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "observable_kind": self.observable_kind,
            "domain": self.domain,
            "A_ref": self.reference_amplitude,
        }

    def tau_context(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "system": self.system_type,
            "tau": self.characteristic_time,
        }

    def power_context(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "system": self.system_type,
            "Pc": self.characteristic_power,
        }

    def activity_context(self) -> Dict[str, Any]:
        """Legacy context retained for serialization compatibility."""
        return {
            "domain": self.domain,
            "mechanism": self.mechanism,
            "A_fact": self.activity_factor,
        }

    def unit_contract(self) -> Dict[str, str]:
        """Describe unit labels attached to canonical physical quantities."""
        return {
            "u": self.unit,
            "A_ref": self.unit,
            "xi": self.coordinate_unit,
            "tau": self.coordinate_unit,
            "P_c": self.power_unit,
            "b": self.agencity_unit,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "observable_kind": self.observable_kind,
            "unit": self.unit,
            "coordinate_unit": self.coordinate_unit,
            "power_unit": self.power_unit,
            "agencity_unit": self.agencity_unit,
            "system_type": self.system_type,
            "mechanism": self.mechanism,
            "tau": self.characteristic_time,
            "Pc": self.characteristic_power,
            "A_ref": self.reference_amplitude,
            "memory_window": self.memory_window,
            "A_fact": self.activity_factor,
            "resolution_scale": self.resolution_scale,
        }

    def __repr__(self) -> str:
        return (
            "ExperimentMetadata("
            f"domain='{self.domain}', observable_kind='{self.observable_kind}', "
            f"unit='{self.unit}', coordinate_unit='{self.coordinate_unit}', "
            f"power_unit='{self.power_unit}', system_type='{self.system_type}', "
            f"tau={self.characteristic_time}, Pc={self.characteristic_power}, "
            f"A_ref={self.reference_amplitude})"
        )
