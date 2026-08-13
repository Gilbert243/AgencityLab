from __future__ import annotations

import numpy as np
import pytest

from agencitylab.extensions import (
    RIEMANNIAN_EXTENSION_STATUS,
    mean_contrast_criterion,
    orientational_entropy_criterion,
    riemannian_dynamic_intensity,
    riemannian_inner_product,
    riemannian_speed,
)
from agencitylab.scientific_status import ScientificStatus


def test_riemannian_extension_remains_experimental_and_intrinsic() -> None:
    assert RIEMANNIAN_EXTENSION_STATUS is ScientificStatus.EXPERIMENTAL
    velocity = np.array([[3.0, 4.0], [1.0, -2.0]])
    acceleration = np.array([[2.0, 0.0], [-1.0, 0.5]])
    metric = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()

    np.testing.assert_allclose(riemannian_speed(velocity, metric), [5.0, np.sqrt(5.0)])
    projection = np.sum(acceleration * velocity, axis=-1)
    expected = np.sqrt(np.sum(velocity**2, axis=-1) + projection**2)
    np.testing.assert_allclose(
        riemannian_dynamic_intensity(velocity, acceleration, metric),
        expected,
    )


def test_riemannian_inner_product_uses_supplied_metric() -> None:
    first = np.array([1.0, 2.0])
    second = np.array([-1.0, 3.0])
    metric = np.array([[4.0, 0.5], [0.5, 2.0]])
    expected = first @ metric @ second
    assert riemannian_inner_product(first, second, metric) == pytest.approx(expected)


def test_riemannian_metric_must_be_symmetric_positive_definite() -> None:
    vector = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        riemannian_speed(vector, np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        riemannian_speed(vector, np.array([[1.0, 0.0], [0.0, -1.0]]))


def test_phi1_mean_contrast_uniform_and_nonuniform_coordinates() -> None:
    contrast = np.array([-1.0, 2.0, -3.0])
    assert mean_contrast_criterion(contrast) == pytest.approx(2.0)

    coordinates = np.array([0.0, 1.0, 3.0])
    expected_integral = 0.5 * (1.0 + 2.0) * 1.0 + 0.5 * (2.0 + 3.0) * 2.0
    assert mean_contrast_criterion(
        contrast,
        coordinates=coordinates,
    ) == pytest.approx(expected_integral / 3.0)


def test_phi3_entropy_uses_explicit_bins_without_epsilon() -> None:
    theta = np.array([-0.75, -0.25, 0.25, 0.75])
    edges = np.array([-1.0, 0.0, 1.0])
    assert orientational_entropy_criterion(theta, bin_edges=edges) == pytest.approx(np.log(2.0))

    concentrated = np.array([-0.8, -0.7, -0.6])
    assert orientational_entropy_criterion(
        concentrated,
        bin_edges=edges,
    ) == pytest.approx(0.0)


def test_phi3_requires_caller_bins_and_can_exclude_undefined_orientation() -> None:
    theta = np.array([-0.5, 0.0, 0.5])
    mask = np.array([True, False, True])
    entropy = orientational_entropy_criterion(
        theta,
        bin_edges=np.array([-1.0, 0.0, 1.0]),
        valid_mask=mask,
    )
    assert entropy == pytest.approx(np.log(2.0))

    with pytest.raises(ValueError):
        orientational_entropy_criterion(
            theta,
            bin_edges=np.array([-1.0, 1.0]),
            valid_mask=np.zeros(3, dtype=bool),
        )
