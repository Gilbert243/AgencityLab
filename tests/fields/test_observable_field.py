from __future__ import annotations

import numpy as np
import pytest

import agencitylab
from agencitylab import AgencityValidationError, PhysicalParameterError, compute_agencity
from agencitylab.fields import ObservableAgencityFieldResult, compute_agencity_field

CANONICAL_FIELDS = (
    "u_star",
    "X_star",
    "A_star",
    "M",
    "O",
    "D",
    "S",
    "J",
    "U",
    "beta",
    "b",
)


def _time(n: int = 64) -> np.ndarray:
    return np.arange(n, dtype=float)


def _signal(t: np.ndarray, phase: float = 0.0) -> np.ndarray:
    return np.sin(0.22 * t + phase) + 0.12 * np.sin(0.51 * t)


def _trajectory(values: np.ndarray, time_axis: int, spatial_index: tuple[int, ...]):
    time_first = np.moveaxis(values, time_axis, 0)
    return time_first[(slice(None), *spatial_index)]


def _assert_scalar_equivalence(
    result: ObservableAgencityFieldResult,
    spatial_index: tuple[int, ...],
    scalar,
) -> None:
    for name in CANONICAL_FIELDS:
        np.testing.assert_allclose(
            _trajectory(getattr(result, name), result.time_axis, spatial_index),
            getattr(scalar, name),
            rtol=0.0,
            atol=0.0,
        )


def test_single_spatial_point_is_exact_scalar_pipeline():
    t = _time()
    u = _signal(t)[:, None]
    field = compute_agencity_field(
        u,
        t,
        A_ref=1.3,
        tau=4.0,
        w=3.0,
        P_c=2.5,
    )
    scalar = compute_agencity(u=u[:, 0], xi=t, A_ref=1.3, tau=4.0, w=3.0, P_c=2.5)

    assert isinstance(field, ObservableAgencityFieldResult)
    assert field.status == "experimental"
    assert field.model == "observable_agencity_field"
    _assert_scalar_equivalence(field, (0,), scalar)


def test_homogeneous_field_repeats_scalar_result():
    t = _time()
    signal = _signal(t)
    u = np.repeat(signal[:, None], 4, axis=1)
    field = compute_agencity_field(u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0)
    scalar = compute_agencity(u=signal, xi=t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0)

    for index in range(4):
        _assert_scalar_equivalence(field, (index,), scalar)


def test_spatial_A_ref_matches_independent_scalar_calls():
    t = _time()
    u = np.column_stack([_signal(t, 0.1 * index) for index in range(3)])
    A_ref = np.array([0.8, 1.2, 2.0])
    field = compute_agencity_field(u, t, A_ref=A_ref, tau=4.0, w=3.0, P_c=1.0)

    for index, local_A_ref in enumerate(A_ref):
        scalar = compute_agencity(
            u=u[:, index], xi=t, A_ref=local_A_ref, tau=4.0, w=3.0, P_c=1.0
        )
        _assert_scalar_equivalence(field, (index,), scalar)


def test_spatial_tau_matches_independent_scalar_calls():
    t = _time()
    u = np.column_stack([_signal(t, 0.2 * index) for index in range(3)])
    tau = np.array([2.0, 4.0, 6.0])
    field = compute_agencity_field(u, t, A_ref=1.0, tau=tau, w=2.0, P_c=1.0)

    for index, local_tau in enumerate(tau):
        scalar = compute_agencity(
            u=u[:, index], xi=t, A_ref=1.0, tau=local_tau, w=2.0, P_c=1.0
        )
        _assert_scalar_equivalence(field, (index,), scalar)


def test_spatial_w_is_independent_from_tau():
    t = _time()
    u = np.column_stack([_signal(t), _signal(t, 0.4), _signal(t, 0.8)])
    tau = np.array([4.0, 4.0, 4.0])
    w = np.array([2.0, 3.0, 5.0])
    field = compute_agencity_field(u, t, A_ref=1.0, tau=tau, w=w, P_c=1.0)

    np.testing.assert_array_equal(field.w, w)
    for index, local_w in enumerate(w):
        scalar = compute_agencity(
            u=u[:, index], xi=t, A_ref=1.0, tau=4.0, w=local_w, P_c=1.0
        )
        _assert_scalar_equivalence(field, (index,), scalar)


def test_unspecified_w_uses_documented_software_fallback_only():
    t = _time()
    tau = np.array([2.0, 5.0])
    u = np.column_stack([_signal(t), _signal(t, 0.3)])
    field = compute_agencity_field(u, t, A_ref=1.0, tau=tau, P_c=1.0)

    np.testing.assert_array_equal(field.w, tau)
    assert field.metadata["w_mode"] == "fallback_w_equals_tau"
    assert field.metadata["w_resolution"] == (
        "w was unspecified; implementation convention w = tau was used"
    )


def test_spatial_power_zeroes_flux_locally_without_zeroing_beta():
    t = _time()
    u = np.column_stack([_signal(t), _signal(t, 0.2), _signal(t, 0.4)])
    power = np.array([0.0, 2.0, 4.0])
    field = compute_agencity_field(u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=power)

    np.testing.assert_array_equal(field.b[:, 0], 0.0j)
    assert np.any(np.abs(field.beta[:, 0]) > 0.0)
    for index, local_power in enumerate(power):
        scalar = compute_agencity(
            u=u[:, index], xi=t, A_ref=1.0, tau=4.0, w=3.0, P_c=local_power
        )
        _assert_scalar_equivalence(field, (index,), scalar)


def test_spacetime_power_obeys_flux_identity_pointwise():
    t = _time()
    u = np.column_stack([_signal(t), _signal(t, 0.5)])
    power = np.column_stack(
        [np.linspace(0.0, 2.0, t.size), np.linspace(3.0, 0.0, t.size)]
    )
    field = compute_agencity_field(u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=power)

    assert field.metadata["P_c_mode"] == "spacetime"
    np.testing.assert_allclose(field.b, power * field.beta, rtol=0.0, atol=0.0)
    for index in range(2):
        scalar = compute_agencity(
            u=u[:, index], xi=t, A_ref=1.0, tau=4.0, w=3.0, P_c=power[:, index]
        )
        _assert_scalar_equivalence(field, (index,), scalar)


def test_constant_local_trajectory_is_exact_rest_without_affecting_neighbor():
    t = _time()
    dynamic = _signal(t)
    u = np.column_stack([np.full_like(t, 7.0), dynamic])
    field = compute_agencity_field(u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0)

    for name in CANONICAL_FIELDS:
        np.testing.assert_array_equal(field.__getattribute__(name)[:, 0], 0)
    scalar = compute_agencity(u=dynamic, xi=t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0)
    _assert_scalar_equivalence(field, (1,), scalar)


@pytest.mark.parametrize("spatial_shape", [(3,), (2, 3), (2, 2, 2)])
def test_generic_spatial_dimensions(spatial_shape):
    t = _time(48)
    base = _signal(t)
    u = np.empty((t.size, *spatial_shape), dtype=float)
    for index in np.ndindex(spatial_shape):
        u[(slice(None), *index)] = base + 0.03 * sum(index)

    axes = tuple(np.linspace(0.0, float(size - 1), size) for size in spatial_shape)
    field = compute_agencity_field(
        u,
        t,
        spatial_axes=axes,
        A_ref=np.ones(spatial_shape),
        tau=4.0,
        w=3.0,
        P_c=1.0,
    )

    assert field.spatial_shape == spatial_shape
    assert field.beta.shape == u.shape
    assert field.b.shape == u.shape
    assert len(field.spatial_axes) == len(spatial_shape)


def test_time_axis_minus_one_is_permutation_equivalent():
    t = _time()
    u_time_first = np.empty((t.size, 2, 3), dtype=float)
    for index in np.ndindex((2, 3)):
        u_time_first[(slice(None), *index)] = _signal(t, 0.1 * sum(index))
    u_time_last = np.moveaxis(u_time_first, 0, -1)

    first = compute_agencity_field(
        u_time_first, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0, time_axis=0
    )
    last = compute_agencity_field(
        u_time_last, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0, time_axis=-1
    )

    for name in CANONICAL_FIELDS:
        np.testing.assert_allclose(
            getattr(first, name), np.moveaxis(getattr(last, name), -1, 0), rtol=0.0, atol=0.0
        )


def test_default_spatial_axes_are_recorded_as_sample_indices():
    t = _time()
    field = compute_agencity_field(
        np.column_stack([_signal(t), _signal(t, 0.3)]),
        t,
        A_ref=1.0,
        tau=4.0,
        w=3.0,
        P_c=1.0,
    )
    np.testing.assert_array_equal(field.spatial_axes[0], np.arange(2.0))
    assert field.metadata["spatial_axes_mode"] == "sample_index"
    assert field.metadata["crm_scope"] == "temporal_only_independent_at_each_spatial_location"


@pytest.mark.parametrize(
    ("u", "t", "kwargs", "error"),
    [
        (np.empty((4, 0)), np.arange(4.0), {}, AgencityValidationError),
        (np.ones((2, 2)), np.arange(2.0), {}, AgencityValidationError),
        (np.array([[0.0, 1.0], [np.nan, 2.0], [1.0, 3.0]]), np.arange(3.0), {}, AgencityValidationError),
        (np.array([[0.0, 1.0], [np.inf, 2.0], [1.0, 3.0]]), np.arange(3.0), {}, AgencityValidationError),
        (np.ones((4, 2)), np.arange(4.0), {"time_axis": 2}, AgencityValidationError),
        (np.ones((4, 2)), np.arange(4.0), {"spatial_axes": (np.arange(3.0),)}, AgencityValidationError),
        (np.ones((4, 3)), np.arange(4.0), {"spatial_axes": (np.array([0.0, 2.0, 1.0]),)}, AgencityValidationError),
        (np.ones((4, 2)), np.arange(4.0), {"A_ref": 0.0}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"tau": 0.0}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"w": 0.0}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"P_c": -1.0}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"A_ref": np.ones(3)}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"tau": np.ones((4, 2))}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"w": np.ones((4, 2))}, PhysicalParameterError),
        (np.ones((4, 2)), np.arange(4.0), {"P_c": np.ones(3)}, PhysicalParameterError),
    ],
)
def test_validation_rejects_invalid_field_contract(u, t, kwargs, error):
    options = dict(A_ref=1.0, tau=1.0, w=1.0, P_c=1.0)
    options.update(kwargs)
    with pytest.raises(error):
        compute_agencity_field(u, t, **options)


def test_local_translation_sign_and_power_invariances_are_inherited():
    t = _time()
    u = np.column_stack([_signal(t), _signal(t, 0.4)])
    base = compute_agencity_field(u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0)
    translated = compute_agencity_field(
        u + 9.0, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0
    )
    inverted = compute_agencity_field(-u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=2.0)
    scaled_power = compute_agencity_field(
        u, t, A_ref=1.0, tau=4.0, w=3.0, P_c=6.0
    )

    np.testing.assert_allclose(translated.beta, base.beta, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(translated.b, base.b, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(inverted.beta, base.beta, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(inverted.b, base.b, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(scaled_power.beta, base.beta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scaled_power.b, 3.0 * base.b, rtol=0.0, atol=2e-15)


def test_top_level_public_exports_exist():
    assert agencitylab.compute_agencity_field is compute_agencity_field
    assert agencitylab.ObservableAgencityFieldResult is ObservableAgencityFieldResult
