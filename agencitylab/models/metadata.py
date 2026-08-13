"""Scientific reproducibility metadata for AgencityLab.

Metadata carries contextual information required to reproduce a computation.
Unit fields are descriptive labels only: AgencityLab never silently converts
magnitudes between unit systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _clean_text(value: Any, *, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _positive_optional(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
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


def _agencity_flux_unit(power_unit: str) -> str:
    return f"{power_unit}·nat" if power_unit else ""


@dataclass(slots=True)
class ExperimentMetadata:
    """Reproducibility metadata for one Agencity computation.

    ``unit`` labels ``u`` and ``A_ref``; ``coordinate_unit`` labels ``xi``,
    ``tau`` and ``w``; ``power_unit`` labels ``P_c``. The agencity flux ``b``
    therefore carries the informational-power label ``power_unit·nat``.

    ``characteristic_power`` stores a scalar physical ``P_c >= 0``. A sampled
    time-varying ``P_c`` remains on :class:`AgencityResult` and is identified in
    ``extra`` rather than being collapsed to a scalar.
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

    reference_amplitude: float | None = None
    characteristic_time: float | None = None
    characteristic_power: float | None = None
    memory_window: float | None = None

    system_type: str = ""
    mechanism: str = ""
    environment: str = ""
    geometry: str = ""
    agencitylab_version: str = ""

    component_units: list[str] = field(default_factory=list)
    component_kinds: list[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

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
            "agencitylab_version",
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
        self.characteristic_power = _nonnegative_optional(
            self.characteristic_power, name="characteristic_power"
        )
        self.memory_window = _positive_optional(self.memory_window, name="memory_window")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe detached dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | ExperimentMetadata | None,
    ) -> ExperimentMetadata:
        """Create metadata while preserving unknown descriptive fields in ``extra``."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return cls(**data.to_dict())
        if not isinstance(data, dict):
            raise ValueError("metadata must be a dictionary, ExperimentMetadata, or None")

        payload = dict(data)
        allowed = {item.name for item in fields(cls)}
        known: dict[str, Any] = {}
        for key in list(payload):
            if key in allowed:
                known[key] = payload.pop(key)

        extra = dict(known.get("extra", {}) or {})
        extra.update(payload)
        known["extra"] = extra
        return cls(**known)

    def with_updates(self, **updates: Any) -> ExperimentMetadata:
        payload = self.to_dict()
        payload.update(updates)
        return ExperimentMetadata.from_dict(payload)

    def has_reference_amplitude(self) -> bool:
        return self.reference_amplitude is not None

    def has_characteristic_time(self) -> bool:
        return self.characteristic_time is not None

    def has_characteristic_power(self) -> bool:
        return self.characteristic_power is not None

    @property
    def agencity_unit(self) -> str:
        """Unit label for the observable agencity flux ``b``."""
        return _agencity_flux_unit(self.power_unit)

    def reference_context(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "observable_kind": self.observable_kind,
            "domain": self.domain,
            "A_ref": self.reference_amplitude,
        }

    def tau_context(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "system": self.system_type,
            "tau": self.characteristic_time,
        }

    def power_context(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "system": self.system_type,
            "P_c": self.characteristic_power,
        }

    def unit_contract(self) -> dict[str, str]:
        """Describe the stable 1.0 unit-label contract."""
        return {
            "u": self.unit,
            "A_ref": self.unit,
            "xi": self.coordinate_unit,
            "tau": self.coordinate_unit,
            "w": self.coordinate_unit,
            "P_c": self.power_unit,
            "b": self.agencity_unit,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "agencitylab_version": self.agencitylab_version,
            "domain": self.domain,
            "observable_kind": self.observable_kind,
            "unit": self.unit,
            "coordinate_unit": self.coordinate_unit,
            "power_unit": self.power_unit,
            "agencity_unit": self.agencity_unit,
            "system_type": self.system_type,
            "mechanism": self.mechanism,
            "tau": self.characteristic_time,
            "P_c": self.characteristic_power,
            "A_ref": self.reference_amplitude,
            "memory_window": self.memory_window,
        }

    def __repr__(self) -> str:
        return (
            "ExperimentMetadata("
            f"domain='{self.domain}', observable_kind='{self.observable_kind}', "
            f"unit='{self.unit}', coordinate_unit='{self.coordinate_unit}', "
            f"power_unit='{self.power_unit}', system_type='{self.system_type}', "
            f"tau={self.characteristic_time}, P_c={self.characteristic_power}, "
            f"A_ref={self.reference_amplitude})"
        )
