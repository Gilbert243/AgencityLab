"""Serialization, freezing and content-addressing of biological protocols."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from agencitylab.exceptions import AgencityValidationError
from agencitylab.models import ParameterProvenance

from .models import (
    BiologicalCapacity,
    BiologicalCoordinate,
    BiologicalMemoryReference,
    BiologicalProtocol,
    BiologicalReference,
    BiologicalTimeReference,
    DatasetSplit,
    FrozenBiologicalProtocol,
    StructuralEpoch,
)
from .validation import validate_protocol


def _payload(protocol: BiologicalProtocol) -> str:
    return json.dumps(
        protocol.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _digest(protocol: BiologicalProtocol) -> str:
    return hashlib.sha256(_payload(protocol).encode("utf-8")).hexdigest()


def freeze_protocol(protocol: BiologicalProtocol) -> FrozenBiologicalProtocol:
    """Validate and freeze a reproducible biological protocol snapshot."""
    validation = validate_protocol(protocol)
    if not validation.valid:
        details = "; ".join(validation.errors)
        raise AgencityValidationError(f"biology protocol is incomplete or invalid: {details}")
    return FrozenBiologicalProtocol(
        protocol=protocol,
        content_hash=_digest(protocol),
        frozen_at=datetime.now(timezone.utc).isoformat(),
    )


def _provenance(data: Mapping[str, Any] | ParameterProvenance) -> ParameterProvenance:
    if isinstance(data, ParameterProvenance):
        return data
    return ParameterProvenance.from_dict(data)


def protocol_from_dict(data: Mapping[str, Any]) -> BiologicalProtocol:
    """Reconstruct a biological protocol from :meth:`BiologicalProtocol.to_dict`."""
    reference_data = data["reference"]
    reference = BiologicalReference(
        taxon=reference_data["taxon"],
        support=reference_data["support"],
        causal_input=reference_data["causal_input"],
        causal_output=reference_data["causal_output"],
        external_conditions=reference_data.get("external_conditions", {}),
        applicability_domain=reference_data.get("applicability_domain", {}),
        identifier=reference_data.get("identifier", ""),
        version=reference_data.get("version", "1"),
    )

    capacity_data = data["capacity"]
    epoch_data = capacity_data["structural_epoch"]
    epoch = StructuralEpoch(
        subject_id=epoch_data["subject_id"],
        start=epoch_data["start"],
        end=epoch_data["end"],
        resolution=epoch_data.get("resolution", ""),
        criteria=tuple(epoch_data.get("criteria", ())),
        evidence=tuple(epoch_data.get("evidence", ())),
    )
    capacity = BiologicalCapacity(
        subject_id=capacity_data["subject_id"],
        value=capacity_data["value"],
        structural_epoch=epoch,
        unit=capacity_data.get("unit", "W"),
        challenge_protocol=capacity_data.get("challenge_protocol", ""),
        measurement_method=capacity_data.get("measurement_method", ""),
        reference_conditions=capacity_data.get("reference_conditions", {}),
        uncertainty=capacity_data.get("uncertainty"),
        repeatability=capacity_data.get("repeatability"),
        provenance=_provenance(capacity_data["provenance"]),
    )

    time_data = data["time_reference"]
    time_reference = BiologicalTimeReference(
        reference=reference,
        q_reference=time_data["q_reference"],
        tau=time_data["tau"],
        sample_size=time_data["sample_size"],
        dispersion=time_data.get("dispersion"),
        unit=time_data.get("unit", "s"),
        source_ids=tuple(time_data.get("source_ids", ())),
        provenance=_provenance(time_data["provenance"]),
    )

    memory_data = data["memory_reference"]
    memory_reference = BiologicalMemoryReference(
        reference=reference,
        w=memory_data["w"],
        individual_causal_times=tuple(memory_data["individual_causal_times"]),
        dispersion=memory_data.get("dispersion"),
        unit=memory_data.get("unit", "s"),
        source_ids=tuple(memory_data.get("source_ids", ())),
        provenance=_provenance(memory_data["provenance"]),
    )

    coordinates = tuple(
        BiologicalCoordinate(
            name=item["name"],
            block=item["block"],
            unit=item["unit"],
            A_ref=item.get("A_ref"),
            description=item.get("description", ""),
            provenance=_provenance(item["provenance"]),
        )
        for item in data["coordinates"]
    )
    split_data = data.get("dataset_split", {})
    dataset_split = DatasetSplit(
        discovery=frozenset(split_data.get("discovery", ())),
        closure=frozenset(split_data.get("closure", ())),
        calibration=frozenset(split_data.get("calibration", ())),
        test=frozenset(split_data.get("test", ())),
    )

    return BiologicalProtocol(
        reference=reference,
        capacity=capacity,
        time_reference=time_reference,
        memory_reference=memory_reference,
        coordinates=coordinates,
        dataset_split=dataset_split,
        measurement_map_id=data.get("measurement_map_id", ""),
        measurement_map_version=data.get("measurement_map_version", ""),
        protocol_id=data.get("protocol_id", "biology"),
        version=data.get("version", "1"),
        metrology_level=data.get("metrology_level"),
        target_informed_calibration=bool(data.get("target_informed_calibration", False)),
        notes=data.get("notes", ""),
    )


def frozen_protocol_to_dict(protocol: FrozenBiologicalProtocol) -> dict[str, Any]:
    """Return a serialization-safe frozen protocol payload."""
    return {
        "protocol": protocol.protocol.to_dict(),
        "content_hash": protocol.content_hash,
        "frozen_at": protocol.frozen_at,
    }


def frozen_protocol_from_dict(data: Mapping[str, Any]) -> FrozenBiologicalProtocol:
    """Reconstruct and verify a serialized frozen biological protocol."""
    protocol = protocol_from_dict(data["protocol"])
    validation = validate_protocol(protocol)
    if not validation.valid:
        raise AgencityValidationError(
            "serialized biology protocol is invalid: " + "; ".join(validation.errors)
        )
    expected = _digest(protocol)
    supplied = str(data["content_hash"]).strip()
    if supplied != expected:
        raise AgencityValidationError("serialized biology protocol content hash does not match")
    return FrozenBiologicalProtocol(
        protocol=protocol,
        content_hash=supplied,
        frozen_at=str(data["frozen_at"]).strip(),
    )
