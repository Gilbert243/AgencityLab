"""
metadata.py

Metadata models for AgencityLab.

Metadata is separated from numerical payloads to preserve:
    - scientific clarity,
    - reproducibility,
    - serialization stability,
    - canonical parameter resolution.

This version supports:
    - A_ref resolution,
    - tau structural resolution,
    - A_fact resolution,
    - physical resolution scale,
    - future multi-domain workflows.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    fields,
)

from datetime import (
    datetime,
    timezone,
)

from typing import (
    Any,
    Dict,
    Optional,
)


# ============================================================
# METADATA MODEL
# ============================================================

@dataclass(slots=True)
class ExperimentMetadata:
    """
    Metadata attached to a signal or analysis result.

    This metadata contains:

    - descriptive information,
    - physical observable information,
    - structural system information,
    - canonical agencity parameters.

    Important
    ---------
    Canonical parameters are structural and must remain
    independent from the observable signal u(t).
    """

    # ========================================================
    # DESCRIPTIVE METADATA
    # ========================================================

    title: str = ""

    description: str = ""

    author: str = ""

    domain: str = ""

    source: str = ""

    tags: list[str] = field(
        default_factory=list
    )

    # ========================================================
    # SIGNAL DESCRIPTION
    # ========================================================

    coordinate_name: str = "t"

    signal_name: str = "u"

    observable_kind: str = ""

    unit: str = ""

    # ========================================================
    # CANONICAL STRUCTURAL PARAMETERS
    # ========================================================

    reference_amplitude: Optional[float] = None
    """
    Canonical A_ref.
    """

    characteristic_time: Optional[float] = None
    """
    Structural tau.
    """

    characteristic_power: Optional[float] = None
    """
    Structural characteristic power Pc.
    """

    activity_factor: Optional[float] = None
    """
    Structural activity factor A_fact.
    """

    # ========================================================
    # PHYSICAL RESOLUTION
    # ========================================================

    resolution_scale: Optional[float] = None
    """
    Physical observation / resolution scale.

    This is NOT arbitrary denoising.

    It represents:
        - instrumental resolution,
        - observational scale,
        - physical coarse-graining,
        - bandwidth limitation.
    """

    # ========================================================
    # SYSTEM PHYSICS
    # ========================================================

    system_type: str = ""
    """
    Generic system category.
    """

    mechanism: str = ""
    """
    Dominant active mechanism.
    """

    environment: str = ""
    """
    Experimental environment.
    """

    geometry: str = ""
    """
    Geometry or topology description.
    """

    # ========================================================
    # MULTIVARIATE SUPPORT
    # ========================================================

    component_units: list[str] = field(
        default_factory=list
    )

    component_kinds: list[str] = field(
        default_factory=list
    )

    # ========================================================
    # SYSTEM INFO
    # ========================================================

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    extra: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to serializable dictionary.
        """

        return asdict(self)

    # ========================================================
    # DESERIALIZATION
    # ========================================================

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any] | None,
    ) -> "ExperimentMetadata":
        """
        Create metadata from dictionary.

        Unknown fields are preserved inside `extra`
        for forward compatibility.
        """

        if data is None:
            return cls()

        payload = dict(data)

        allowed = {
            f.name
            for f in fields(cls)
        }

        # ====================================================
        # KNOWN FIELDS
        # ====================================================

        known = {}

        for key in list(payload.keys()):

            if key in allowed:

                known[key] = payload.pop(key)

        # ====================================================
        # EXTRA FIELDS
        # ====================================================

        extra = dict(
            known.get("extra", {})
        )

        extra.update(payload)

        known["extra"] = extra

        return cls(**known)

    # ========================================================
    # HELPERS
    # ========================================================

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

    # ========================================================
    # CANONICAL CONTEXT EXPORTS
    # ========================================================

    def reference_context(self) -> Dict[str, Any]:
        """
        Context used for A_ref resolution.
        """

        return {
            "unit": self.unit,
            "observable_kind": self.observable_kind,
            "domain": self.domain,
            "A_ref": self.reference_amplitude,
        }

    def tau_context(self) -> Dict[str, Any]:
        """
        Context used for tau resolution.
        """

        return {
            "domain": self.domain,
            "system": self.system_type,
            "tau": self.characteristic_time,
        }

    def power_context(self) -> Dict[str, Any]:
        """
        Context used for Pc resolution.
        """

        return {
            "domain": self.domain,
            "system": self.system_type,
            "Pc": self.characteristic_power,
        }

    def activity_context(self) -> Dict[str, Any]:
        """
        Context used for A_fact resolution.
        """

        return {
            "domain": self.domain,
            "mechanism": self.mechanism,
            "A_fact": self.activity_factor,
        }

    # ========================================================
    # SCIENTIFIC SUMMARY
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        """
        Compact scientific metadata summary.
        """

        return {

            "domain": self.domain,

            "observable_kind": self.observable_kind,

            "unit": self.unit,

            "system_type": self.system_type,

            "mechanism": self.mechanism,

            "tau": self.characteristic_time,

            "Pc": self.characteristic_power,

            "A_ref": self.reference_amplitude,

            "A_fact": self.activity_factor,

            "resolution_scale": self.resolution_scale,
        }

    # ========================================================
    # STRING REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:

        return (
            "ExperimentMetadata("
            f"domain='{self.domain}', "
            f"observable_kind='{self.observable_kind}', "
            f"unit='{self.unit}', "
            f"system_type='{self.system_type}', "
            f"mechanism='{self.mechanism}', "
            f"tau={self.characteristic_time}, "
            f"Pc={self.characteristic_power}, "
            f"A_ref={self.reference_amplitude}, "
            f"A_fact={self.activity_factor}"
            ")"
        )