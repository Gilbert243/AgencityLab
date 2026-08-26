"""Experimental biological mapping for the Theory of Agencity.

The namespace prepares biological references, metrology and protocols, then
orchestrates the existing public AgencityLab computation APIs. It does not
redefine the canonical equations and does not validate or invalidate the theory.
"""

from agencitylab.scientific_status import ScientificStatus

from .computation import compute_biological_agencity
from .metrology import BiologicalMeasurementMap, reconstruct_observable
from .models import (
    BiologicalAgencityResult,
    BiologicalCapacity,
    BiologicalCoordinate,
    BiologicalMeasurement,
    BiologicalMemoryReference,
    BiologicalObservable,
    BiologicalProtocol,
    BiologicalReference,
    BiologicalResolution,
    BiologicalTimeReference,
    DatasetRole,
    DatasetSplit,
    FrozenBiologicalProtocol,
    MetrologyLevel,
    ObservableBlock,
    ScientificValidation,
    StructuralEpoch,
)
from .protocols import (
    freeze_protocol,
    frozen_protocol_from_dict,
    frozen_protocol_to_dict,
    protocol_from_dict,
)
from .registry import BiologicalReferenceBundle, BiologicalReferenceRegistry
from .timescales import (
    biological_reference_time,
    calibrate_memory_reference,
    causal_time,
    specific_metabolic_reference,
)
from .validation import (
    validate_observable_against_protocol,
    validate_protocol,
    validate_reference_amplitudes,
)

SCIENTIFIC_STATUS = ScientificStatus.EXPERIMENTAL

__all__ = [
    "SCIENTIFIC_STATUS",
    "ObservableBlock",
    "MetrologyLevel",
    "DatasetRole",
    "BiologicalReference",
    "StructuralEpoch",
    "BiologicalCapacity",
    "BiologicalTimeReference",
    "BiologicalMemoryReference",
    "BiologicalResolution",
    "BiologicalCoordinate",
    "BiologicalMeasurement",
    "BiologicalMeasurementMap",
    "BiologicalObservable",
    "DatasetSplit",
    "BiologicalProtocol",
    "FrozenBiologicalProtocol",
    "ScientificValidation",
    "BiologicalAgencityResult",
    "BiologicalReferenceBundle",
    "BiologicalReferenceRegistry",
    "specific_metabolic_reference",
    "biological_reference_time",
    "causal_time",
    "calibrate_memory_reference",
    "reconstruct_observable",
    "validate_reference_amplitudes",
    "validate_protocol",
    "validate_observable_against_protocol",
    "freeze_protocol",
    "protocol_from_dict",
    "frozen_protocol_to_dict",
    "frozen_protocol_from_dict",
    "compute_biological_agencity",
]
