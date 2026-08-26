import numpy as np
import pytest

import agencitylab.biology as bio


def _coordinates():
    return (
        bio.BiologicalCoordinate("mass_state", bio.ObservableBlock.Q, "kg", A_ref=1.0),
        bio.BiologicalCoordinate("flux_state", bio.ObservableBlock.PHI, "W", A_ref=2.0),
    )


def test_measurement_and_observable_are_distinct_contracts():
    measurement = bio.BiologicalMeasurement(
        xi=[0.0, 1.0, 2.0],
        values=[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
        channels=("sensor-a", "sensor-b"),
        units=("V", "V"),
        instrument_id="instrument-1",
        uncertainty=[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
        covariance=np.diag([0.01, 0.04]),
    )
    mapping = bio.BiologicalMeasurementMap(
        identifier="cal-map",
        version="1",
        input_channels=("sensor-a", "sensor-b"),
        input_units=("V", "V"),
        output_coordinates=_coordinates(),
        matrix=[[2.0, 0.0], [0.0, 3.0]],
        offset=[1.0, -1.0],
    )
    observable = bio.reconstruct_observable(measurement, mapping)
    np.testing.assert_allclose(
        observable.values,
        [[3.0, 5.0], [5.0, 11.0], [7.0, 17.0]],
    )
    assert observable.measurement_map_id == "cal-map"
    assert observable.measurement_map_version == "1"
    assert observable.covariance.shape == (2, 2)
    np.testing.assert_allclose(observable.covariance, np.diag([0.04, 0.36]))


def test_uncertainty_does_not_silently_assume_independent_covariance():
    measurement = bio.BiologicalMeasurement(
        xi=[0.0, 1.0],
        values=[[1.0, 2.0], [2.0, 3.0]],
        channels=("sensor-a", "sensor-b"),
        units=("V", "V"),
        instrument_id="instrument-1",
        uncertainty=[[0.1, 0.2], [0.1, 0.2]],
    )
    mapping = bio.BiologicalMeasurementMap(
        identifier="cal-map",
        version="1",
        input_channels=("sensor-a", "sensor-b"),
        input_units=("V", "V"),
        output_coordinates=_coordinates(),
        matrix=np.eye(2),
    )
    observable = bio.reconstruct_observable(measurement, mapping)
    assert observable.covariance is None


def test_measurement_map_requires_exact_calibrated_channels():
    measurement = bio.BiologicalMeasurement(
        xi=[0.0, 1.0],
        values=[[1.0, 2.0], [2.0, 3.0]],
        channels=("wrong", "sensor-b"),
        units=("V", "V"),
        instrument_id="instrument-1",
    )
    mapping = bio.BiologicalMeasurementMap(
        identifier="cal-map",
        version="1",
        input_channels=("sensor-a", "sensor-b"),
        input_units=("V", "V"),
        output_coordinates=_coordinates(),
        matrix=np.eye(2),
    )
    with pytest.raises(ValueError, match="do not match"):
        bio.reconstruct_observable(measurement, mapping)


def test_measurement_map_requires_exact_calibrated_units():
    measurement = bio.BiologicalMeasurement(
        xi=[0.0, 1.0],
        values=[[1.0, 2.0], [2.0, 3.0]],
        channels=("sensor-a", "sensor-b"),
        units=("mV", "V"),
        instrument_id="instrument-1",
    )
    mapping = bio.BiologicalMeasurementMap(
        identifier="cal-map",
        version="1",
        input_channels=("sensor-a", "sensor-b"),
        input_units=("V", "V"),
        output_coordinates=_coordinates(),
        matrix=np.eye(2),
    )
    with pytest.raises(ValueError, match="units do not match"):
        bio.reconstruct_observable(measurement, mapping)


def test_observable_covariance_shape_is_explicit():
    with pytest.raises(ValueError, match=r"shape \(m,m\)"):
        bio.BiologicalObservable(
            xi=[0.0, 1.0],
            values=[[1.0, 2.0], [2.0, 3.0]],
            coordinates=_coordinates(),
            covariance=np.eye(3),
        )


def test_covariance_must_be_symmetric_positive_semidefinite():
    with pytest.raises(ValueError, match="symmetric"):
        bio.BiologicalMeasurement(
            xi=np.array([0.0, 1.0]),
            values=np.array([[1.0, 2.0], [2.0, 3.0]]),
            channels=("a", "b"),
            units=("V", "V"),
            instrument_id="instrument",
            covariance=np.array([[1.0, 2.0], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        bio.BiologicalMeasurement(
            xi=np.array([0.0, 1.0]),
            values=np.array([[1.0, 2.0], [2.0, 3.0]]),
            channels=("a", "b"),
            units=("V", "V"),
            instrument_id="instrument",
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )
