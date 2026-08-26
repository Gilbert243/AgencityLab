import dataclasses

import pytest

import agencitylab.biology as bio
from agencitylab.exceptions import AgencityValidationError


def _reference():
    return bio.BiologicalReference(
        taxon="species-x",
        support="whole organism",
        causal_input="challenge",
        causal_output="response",
        identifier="ref-x",
    )


def _protocol(*, unresolved=False, target_informed=False, overlap=False):
    reference = _reference()
    epoch = bio.StructuralEpoch("subject-1", 0.0, 10.0, criteria=("stable structure",))
    capacity = bio.BiologicalCapacity(
        "subject-1",
        120.0,
        epoch,
        challenge_protocol="sustainable frontier protocol",
        measurement_method="indirect calorimetry",
    )
    tau = bio.BiologicalTimeReference(reference, q_reference=10.0, tau=0.1, sample_size=5)
    memory = bio.BiologicalMemoryReference(
        reference,
        w=0.2,
        individual_causal_times=(0.1, 0.2, 0.3),
    )
    coordinates = (
        bio.BiologicalCoordinate(
            "q1",
            bio.ObservableBlock.Q,
            "kg",
            A_ref=None if unresolved else 1.0,
        ),
    )
    split = bio.DatasetSplit(
        closure=frozenset({"a"}),
        test=frozenset({"a" if overlap else "b"}),
    )
    return bio.BiologicalProtocol(
        reference=reference,
        capacity=capacity,
        time_reference=tau,
        memory_reference=memory,
        coordinates=coordinates,
        dataset_split=split,
        measurement_map_id="map",
        measurement_map_version="1",
        protocol_id="protocol-x",
        target_informed_calibration=target_informed,
    )


def test_valid_protocol_freezes_to_stable_content_hash():
    protocol = _protocol()
    first = bio.freeze_protocol(protocol)
    second = bio.freeze_protocol(protocol)
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64
    assert bio.validate_protocol(protocol).valid


def test_frozen_scientific_objects_are_immutable():
    frozen = bio.freeze_protocol(_protocol())
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.protocol.version = "2"


def test_unresolved_A_ref_is_reported_not_inferred():
    protocol = _protocol(unresolved=True)
    validation = bio.validate_protocol(protocol)
    assert not validation.valid
    assert any(item.startswith("unresolved_A_ref") for item in validation.errors)
    with pytest.raises(AgencityValidationError, match="unresolved_A_ref"):
        bio.freeze_protocol(protocol)


def test_target_informed_calibration_is_rejected():
    validation = bio.validate_protocol(_protocol(target_informed=True))
    assert not validation.valid
    assert "target_informed_calibration_is_not_admissible" in validation.errors


def test_closure_test_overlap_is_rejected():
    validation = bio.validate_protocol(_protocol(overlap=True))
    assert not validation.valid
    assert any(item.startswith("dataset_independence_violation") for item in validation.errors)


def test_tau_and_w_may_be_equal_but_are_not_identified():
    protocol = _protocol()
    equal_memory = bio.BiologicalMemoryReference(
        protocol.reference,
        w=protocol.time_reference.tau,
        individual_causal_times=(protocol.time_reference.tau,),
    )
    protocol = dataclasses.replace(protocol, memory_reference=equal_memory)
    validation = bio.validate_protocol(protocol)
    assert validation.valid
    assert "tau_and_w_are_numerically_equal_but_remain_distinct_references" in validation.warnings


def test_tau_and_w_units_must_match():
    protocol = _protocol()
    incompatible_memory = dataclasses.replace(protocol.memory_reference, unit="min")
    protocol = dataclasses.replace(protocol, memory_reference=incompatible_memory)
    validation = bio.validate_protocol(protocol)
    assert not validation.valid
    assert "tau_and_w_time_units_do_not_match" in validation.errors


def test_protocol_roundtrip_preserves_content_hash():
    original = bio.freeze_protocol(_protocol())
    payload = bio.frozen_protocol_to_dict(original)
    restored = bio.frozen_protocol_from_dict(payload)
    assert restored.protocol == original.protocol
    assert restored.content_hash == original.content_hash


def test_serialized_protocol_hash_detects_tampering():
    original = bio.freeze_protocol(_protocol())
    payload = bio.frozen_protocol_to_dict(original)
    payload["protocol"]["capacity"]["value"] = 999.0
    with pytest.raises(AgencityValidationError, match="content hash does not match"):
        bio.frozen_protocol_from_dict(payload)


def test_reference_calibration_sources_cannot_leak_into_test_set():
    protocol = _protocol()
    tau = dataclasses.replace(
        protocol.time_reference,
        source_ids=("t1", "t2", "test-subject", "t4", "t5"),
    )
    memory = dataclasses.replace(
        protocol.memory_reference,
        source_ids=("m1", "m2", "test-subject"),
    )
    split = dataclasses.replace(protocol.dataset_split, test=frozenset({"test-subject"}))
    protocol = dataclasses.replace(
        protocol,
        time_reference=tau,
        memory_reference=memory,
        dataset_split=split,
    )
    validation = bio.validate_protocol(protocol)
    assert not validation.valid
    assert any(item.startswith("tau_reference_uses_test_samples") for item in validation.errors)
    assert any(item.startswith("w_reference_uses_test_samples") for item in validation.errors)
