import numpy as np
import pytest

from agencitylab import AgencityResult, ExperimentMetadata, compute_agencity
from agencitylab.api import (
    AgencityStream,
    BatchItemError,
    StreamNotReadyError,
    StreamStateError,
    pipeline,
    run_batch,
)


def _signal():
    xi = np.arange(8.0)
    u = np.sin(xi)
    return xi, u


def test_compute_accepts_metadata_model_and_preserves_unit_contract():
    xi, u = _signal()
    metadata = ExperimentMetadata(
        unit="V",
        coordinate_unit="s",
        power_unit="W",
        reference_amplitude=2.0,
        characteristic_time=2.0,
        characteristic_power=3.0,
        observable_kind="voltage",
        domain="electronics",
    )

    result = compute_agencity(u, xi, metadata=metadata)

    assert result.A_ref == 2.0
    assert result.tau == 2.0
    assert result.P_c == 3.0
    assert result.memory_window == result.tau
    assert result.metadata.extra["memory_window"] == result.tau
    assert result.unit == "V"
    assert result.coordinate_unit == "s"
    assert result.power_unit == "W"
    assert result.A_ref_unit == "V"
    assert result.tau_unit == "s"
    assert result.b_unit == "W·nat"
    assert result.metadata.unit_contract() == {
        "u": "V",
        "A_ref": "V",
        "xi": "s",
        "tau": "s",
        "w": "s",
        "P_c": "W",
        "b": "W·nat",
    }


def test_generated_coordinate_is_labeled_as_samples():
    _, u = _signal()
    result = compute_agencity(u, A_ref=1.0, tau=2.0, P_c=1.0)
    assert result.coordinate_unit == "sample"
    np.testing.assert_array_equal(result.xi, np.arange(len(u), dtype=float))


def test_compute_rejects_multidimensional_inputs():
    xi, u = _signal()
    with pytest.raises(Exception, match="one-dimensional"):
        compute_agencity(
            np.column_stack([u, u]),
            xi,
            A_ref=1.0,
            tau=2.0,
            P_c=1.0,
        )


def test_pre1_compute_aliases_are_not_part_of_1_0_signature():
    xi, u = _signal()
    with pytest.raises(TypeError):
        compute_agencity(data=u, xi=xi, A_ref=1.0, tau=2.0, P_c=1.0)
    with pytest.raises(TypeError):
        compute_agencity(u, xi, A_ref=1.0, tau=2.0, Pc=4.0)
    with pytest.raises(TypeError):
        compute_agencity(u, xi, A_ref=1.0, tau=2.0, P_c=1.0, made_up_option=True)


def test_explicit_time_varying_characteristic_power_is_preserved_exactly():
    xi, u = _signal()
    P_c = np.linspace(1.0, 2.0, len(xi))
    result = compute_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        P_c=P_c,
        power_unit="W",
    )

    np.testing.assert_array_equal(result.P_c, P_c)
    np.testing.assert_allclose(result.b, P_c * result.beta)
    np.testing.assert_allclose(result.eta, result.beta_abs, rtol=1e-14, atol=0.0)
    assert result.is_time_varying_power
    assert result.metadata.characteristic_power is None
    assert result.metadata.extra["characteristic_power_mode"] == "time_varying"
    assert result.summary()["P_c_mean"] == pytest.approx(np.mean(P_c))
    assert result.summary()["P_c"] is None


def test_callable_characteristic_power_is_evaluated_on_xi():
    xi, u = _signal()
    result = compute_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        P_c=lambda t: 2.0 + 0.1 * t,
    )
    expected = 2.0 + 0.1 * xi
    np.testing.assert_allclose(result.P_c, expected)
    np.testing.assert_allclose(result.b, expected * result.beta)


def test_result_uses_canonical_wrapped_theta_not_analysis_unwrapping():
    xi = np.arange(3.0)
    angles = np.array([3.0, -3.0, 0.0])
    U = np.exp(1j * angles)
    zeros = np.zeros(3)
    result = AgencityResult(
        xi=xi,
        u=zeros,
        u_star=zeros,
        X_star=zeros,
        A_star=zeros,
        t_star=xi,
        tau=1.0,
        P_c=2.0,
        A_ref=1.0,
        M=zeros,
        O=zeros,
        D=zeros,
        S=zeros,
        J=zeros,
        U=U,
        beta=np.zeros(3, dtype=complex),
        b=np.zeros(3, dtype=complex),
    )
    np.testing.assert_allclose(result.theta, np.angle(U))
    assert result.theta[1] < 0.0


def test_result_serialization_roundtrip_uses_strict_1_0_schema():
    xi, u = _signal()
    power = np.linspace(2.0, 3.0, len(xi))
    result = compute_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        P_c=power,
        unit="rad",
        coordinate_unit="s",
        power_unit="W",
    )
    payload = result.to_dict()
    assert payload["schema_version"] == "1.0"

    restored = AgencityResult.from_dict(payload)
    np.testing.assert_allclose(restored.P_c, power)
    np.testing.assert_allclose(restored.beta, result.beta)
    np.testing.assert_allclose(restored.b, result.b)
    np.testing.assert_allclose(restored.theta, result.theta)
    assert restored.unit == "rad"
    assert restored.coordinate_unit == "s"
    assert restored.power_unit == "W"
    assert restored.b_unit == "W·nat"

    development_payload = dict(payload)
    development_payload["schema_version"] = "0.3"
    with pytest.raises(ValueError, match="unsupported result schema"):
        AgencityResult.from_dict(development_payload)


def test_result_payload_missing_required_physics_is_rejected():
    xi, u = _signal()
    result = compute_agencity(u, xi, A_ref=2.0, tau=2.0, P_c=3.0)
    payload = result.to_dict()
    payload.pop("P_c")
    with pytest.raises(ValueError, match="missing required field"):
        AgencityResult.from_dict(payload)


def test_metadata_preserves_unknown_descriptive_fields_in_extra():
    metadata = ExperimentMetadata.from_dict({"unit": "V", "future_field": 42})
    assert metadata.unit == "V"
    assert metadata.extra["future_field"] == 42


def test_batch_supports_per_item_physical_parameters_and_preserves_order():
    xi, u = _signal()
    items = [
        {"xi": xi, "u": u, "P_c": 1.0},
        {"xi": xi, "u": u, "P_c": 2.0, "metadata": {"title": "second"}},
    ]
    results = run_batch(
        items,
        metadata={
            "reference_amplitude": 1.0,
            "characteristic_time": 2.0,
            "unit": "rad",
        },
    )

    assert [result.P_c for result in results] == [1.0, 2.0]
    assert results[1].metadata.title == "second"
    assert results[1].unit == "rad"
    np.testing.assert_allclose(results[1].b, 2.0 * results[0].b)


def test_batch_rejects_pre1_data_item_key():
    xi, u = _signal()
    with pytest.raises(BatchItemError, match="missing 'u'"):
        run_batch(
            [{"xi": xi, "data": u}],
            A_ref=1.0,
            tau=2.0,
            P_c=1.0,
        )


def test_batch_errors_identify_the_failing_item():
    xi, u = _signal()
    with pytest.raises(BatchItemError, match="batch item 1"):
        run_batch(
            [(xi, u), {"xi": xi}],
            A_ref=1.0,
            tau=2.0,
            P_c=1.0,
        )


def test_stream_implicit_coordinates_continue_across_chunks():
    stream = AgencityStream(
        analyze=False,
        A_ref=1.0,
        tau=1.0,
        P_c=1.0,
    )
    stream.update(np.array([0.0, 1.0, 0.0, -1.0]))
    result = stream.update(np.array([0.0, 1.0, 0.0, -1.0]))

    np.testing.assert_array_equal(result.xi, np.arange(8.0))
    assert result.coordinate_unit == "sample"
    assert stream.snapshot()["buffer_length"] == 8


def test_stream_rejects_overlapping_explicit_coordinates():
    stream = AgencityStream(analyze=False, A_ref=1.0, tau=1.0, P_c=1.0)
    stream.update(np.array([0.0, 1.0, 0.0, -1.0]), np.arange(4.0))
    with pytest.raises(StreamStateError, match="strictly after"):
        stream.update(
            np.array([0.0, 1.0, 0.0, -1.0]),
            np.arange(3.0, 7.0),
        )


def test_stream_reports_not_ready_for_incomplete_crm_history():
    stream = AgencityStream(analyze=False, A_ref=1.0, tau=2.0, P_c=1.0)
    with pytest.raises(StreamNotReadyError, match="two complete CRM windows"):
        stream.update(np.array([0.0, 1.0, 0.0]))
    assert stream.snapshot()["buffer_length"] == 3


def test_pipeline_uses_explicit_physical_setters():
    xi, u = _signal()
    pipe = (
        pipeline()
        .from_arrays(xi, u)
        .set_reference_amplitude(1.0)
        .set_characteristic_time(2.0)
        .set_memory_window(2.0)
        .set_characteristic_power(5.0)
        .set_unit("rad")
        .set_coordinate_unit("s")
        .set_power_unit("W")
        .compute()
    )

    assert pipe.result is not None
    assert pipe.result.tau == 2.0
    assert pipe.result.memory_window == 2.0
    assert pipe.result.P_c == 5.0
    assert pipe.result.b_unit == "W·nat"
