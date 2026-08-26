import numpy as np
import pytest

import agencitylab.biology as bio
from agencitylab.exceptions import PhysicalParameterError
from agencitylab.models import AgencityResult


def _frozen_protocol(n_components=1):
    reference = bio.BiologicalReference(
        taxon="species-x",
        support="whole organism",
        causal_input="challenge",
        causal_output="response",
        identifier="ref-x",
    )
    epoch = bio.StructuralEpoch("subject-1", 0.0, 10.0)
    capacity = bio.BiologicalCapacity(
        "subject-1",
        100.0,
        epoch,
        challenge_protocol="frontier",
        measurement_method="method",
    )
    tau = bio.BiologicalTimeReference(reference, 10.0, 0.1, 3)
    memory = bio.BiologicalMemoryReference(reference, 0.2, (0.1, 0.2, 0.3))
    blocks = [bio.ObservableBlock.Q, bio.ObservableBlock.PHI]
    coordinates = tuple(
        bio.BiologicalCoordinate(f"u{k}", blocks[k % 2], "unit", A_ref=float(k + 1))
        for k in range(n_components)
    )
    protocol = bio.BiologicalProtocol(
        reference,
        capacity,
        tau,
        memory,
        coordinates,
        measurement_map_id="map",
        measurement_map_version="1",
    )
    return bio.freeze_protocol(protocol)


def _observable(protocol):
    n = len(protocol.protocol.coordinates)
    xi = np.arange(64, dtype=float) * 0.05
    values = np.column_stack([np.sin(2.0 * (k + 1) * xi) for k in range(n)])
    return bio.BiologicalObservable(
        xi,
        values,
        protocol.protocol.coordinates,
        measurement_map_id="map",
        measurement_map_version="1",
    )


def test_scalar_computation_delegates_with_explicit_biology_parameters():
    protocol = _frozen_protocol(1)
    result = bio.compute_biological_agencity(_observable(protocol), protocol)
    assert result.mode == "scalar"
    assert isinstance(result.result, AgencityResult)
    canonical = result.result
    assert canonical.P_c == 100.0
    assert canonical.tau == 0.1
    assert canonical.A_ref == 1.0
    assert canonical.domain == "biology"
    assert canonical.coordinate_unit == "s"
    assert canonical.power_unit == "W"
    assert canonical.observable_kind == "Q"
    assert canonical.metadata.memory_window == 0.2
    assert canonical.metadata.extra["biology_protocol_hash"] == protocol.content_hash
    assert canonical.b.shape == canonical.xi.shape
    assert np.all(np.isfinite(canonical.b))


def test_computation_rejects_sampling_coarser_than_frozen_w():
    protocol = _frozen_protocol(1)
    coordinate = protocol.protocol.coordinates[0]
    xi = np.arange(8, dtype=float)
    observable = bio.BiologicalObservable(
        xi,
        np.sin(0.2 * xi),
        (coordinate,),
        measurement_map_id="map",
        measurement_map_version="1",
    )
    with pytest.raises(ValueError, match="smaller than one sampling interval"):
        bio.compute_biological_agencity(observable, protocol)


def test_multicoordinate_does_not_broadcast_organism_Pc():
    protocol = _frozen_protocol(2)
    with pytest.raises(PhysicalParameterError, match="not broadcast"):
        bio.compute_biological_agencity(_observable(protocol), protocol)


def test_multicoordinate_requires_partition_sum_to_organism_Pc():
    protocol = _frozen_protocol(2)
    with pytest.raises(PhysicalParameterError, match="sum equals organism P_c"):
        bio.compute_biological_agencity(
            _observable(protocol),
            protocol,
            P_c_components=[60.0, 60.0],
        )


def test_multicoordinate_delegates_explicit_physical_partition():
    protocol = _frozen_protocol(2)
    result = bio.compute_biological_agencity(
        _observable(protocol),
        protocol,
        P_c_components=[40.0, 60.0],
    )
    assert result.mode == "multivariate"
    assert result.P_c_components == (40.0, 60.0)
    multivariate = result.result
    assert multivariate["n_components"] == 2
    np.testing.assert_allclose(multivariate["A_ref"], [1.0, 2.0])
    np.testing.assert_allclose(multivariate["tau"], [0.1, 0.1])
    np.testing.assert_allclose(multivariate["w"], [0.2, 0.2])
    np.testing.assert_allclose(multivariate["P_c_components"][0], 40.0)
    np.testing.assert_allclose(multivariate["P_c_components"][1], 60.0)
    np.testing.assert_allclose(multivariate["P_c_total"], 100.0)
    np.testing.assert_allclose(
        multivariate["b_total"],
        np.sum(multivariate["b_components"], axis=0),
    )
    mask = multivariate["beta_multi_defined"]
    np.testing.assert_allclose(
        multivariate["beta_multi"][mask] * multivariate["P_c_total"][mask],
        multivariate["b_total"][mask],
    )
