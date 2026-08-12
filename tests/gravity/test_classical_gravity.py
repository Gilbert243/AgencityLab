"""Tests for the isolated classical Agencity gravity research contract."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import agencitylab.gravity as gravity
from agencitylab.fields.dynamics import klein_gordon_acceleration
from agencitylab.fields.numerics import UniformRectilinearGrid, laplacian
from agencitylab.fields.physics import QuarticAgencityPotential, vacuum_state
from agencitylab.gravity import (
    CONFORMAL_COUPLING_4D,
    GRAVITY_METRIC_SIGNATURE,
    MINIMAL_COUPLING,
    conformal_coupling,
    covariant_scalar_derivative,
    curved_field_residual,
    einstein_equation_residual,
    einstein_hilbert_density,
    matter_action_density,
    matter_lagrangian_density,
    metric_with_perturbation,
    minimal_coupling,
    minkowski_box,
    minkowski_inverse_metric,
    minkowski_metric,
    nonminimal_coupling_density,
    sqrt_minus_g,
    stress_energy_tensor,
    total_gravity_field_lagrangian_density,
)
from agencitylab.scientific_status import ScientificStatus


def test_gravity_status_and_signature_are_explicit():
    assert gravity.SCIENTIFIC_STATUS is ScientificStatus.RESEARCH
    assert GRAVITY_METRIC_SIGNATURE == (-1, 1, 1, 1)
    np.testing.assert_array_equal(minkowski_metric(), np.diag([-1.0, 1.0, 1.0, 1.0]))


def test_minkowski_inverse_and_sqrt_minus_g():
    metric = minkowski_metric()
    inverse = minkowski_inverse_metric()
    np.testing.assert_allclose(metric @ inverse, np.eye(4))
    assert sqrt_minus_g(metric) == pytest.approx(1.0)


def test_metric_perturbation_zero_limit_is_minkowski():
    eta = minkowski_metric()
    result = metric_with_perturbation(eta, np.zeros((4, 4)))
    np.testing.assert_array_equal(result, eta)


def test_metric_validation_rejects_wrong_shape_and_non_lorentzian_determinant():
    with pytest.raises(ValueError, match="shape"):
        sqrt_minus_g(np.eye(3))
    with pytest.raises(ValueError, match="negative determinant"):
        sqrt_minus_g(np.eye(4))
    with pytest.raises(ValueError, match="finite"):
        sqrt_minus_g(np.full((4, 4), np.nan))


def test_minimal_and_conformal_coupling_are_named_not_universal_defaults():
    assert minimal_coupling() == 0.0
    assert MINIMAL_COUPLING == 0.0
    assert conformal_coupling() == pytest.approx(1.0 / 6.0)
    assert CONFORMAL_COUPLING_4D == pytest.approx(1.0 / 6.0)


def test_external_gauge_derivative_zero_and_nonzero():
    phi = np.array([1.0 + 2.0j, -0.5j])
    partial = np.array(
        [
            [1.0 + 0.5j, 2.0, 0.0, -1.0j],
            [0.5, -1.0j, 2.0 + 1.0j, 0.25],
        ]
    )
    zero_gauge = np.zeros(4)
    np.testing.assert_allclose(
        covariant_scalar_derivative(phi, partial, gauge_field=zero_gauge), partial
    )

    gauge = np.array([0.1, -0.2, 0.3, 0.4])
    expected = partial - 1j * gauge[None, :] * phi[:, None]
    np.testing.assert_allclose(
        covariant_scalar_derivative(phi, partial, gauge_field=gauge), expected
    )


def test_external_gauge_rejects_ambiguous_shape():
    phi = np.ones((2, 3), dtype=complex)
    partial = np.zeros((2, 3, 4), dtype=complex)
    with pytest.raises(ValueError, match="gauge_field"):
        covariant_scalar_derivative(phi, partial, gauge_field=np.ones((3, 4)))


def test_matter_lagrangian_matches_explicit_metric_contraction():
    potential = QuarticAgencityPotential(lambda_=1.2, mu=0.8)
    phi = np.array([0.4 + 0.3j, -0.2 + 0.1j])
    derivatives = np.array(
        [
            [1.0 + 0.2j, 0.5j, 0.0, -0.25],
            [-0.4j, 0.3 + 0.2j, 0.1, 0.0],
        ]
    )
    inverse = minkowski_inverse_metric()
    contraction = np.einsum(
        "mn,...m,...n->...", inverse, np.conjugate(derivatives), derivatives
    )
    expected = 0.5 * contraction - potential.value(phi)
    result = matter_lagrangian_density(phi, derivatives, inverse, potential)
    np.testing.assert_allclose(result, expected)
    assert np.isrealobj(result)


def test_matter_action_and_total_action_minimal_flat_limit():
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    phi = np.array([0.0j, 0.25 + 0.1j])
    derivatives = np.zeros(phi.shape + (4,), dtype=complex)
    measure = np.ones(phi.shape)
    curvature = np.zeros(phi.shape)
    matter = matter_action_density(
        phi, derivatives, minkowski_inverse_metric(), measure, potential
    )
    total = total_gravity_field_lagrangian_density(
        phi,
        derivatives,
        minkowski_inverse_metric(),
        measure,
        curvature,
        potential,
        xi=0.0,
        gravitational_constant=1.0,
    )
    np.testing.assert_allclose(total, matter)


def test_zero_curvature_removes_nonminimal_and_einstein_hilbert_terms():
    phi = np.array([0.0, 1.0 + 2.0j])
    zero_curvature = np.zeros(2)
    np.testing.assert_array_equal(
        nonminimal_coupling_density(phi, 1.0, zero_curvature, xi=1.0 / 6.0),
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        einstein_hilbert_density(1.0, zero_curvature, gravitational_constant=2.0),
        np.zeros(2),
    )


def test_phi_zero_has_no_nan_and_needs_no_epsilon():
    potential = QuarticAgencityPotential(lambda_=2.0, mu=1.0)
    phi = np.zeros(5, dtype=complex)
    residual = curved_field_residual(
        np.zeros_like(phi), phi, potential, scalar_curvature=3.0, xi=1.0 / 6.0
    )
    np.testing.assert_array_equal(residual, np.zeros_like(phi))
    assert np.all(np.isfinite(residual))


def test_homogeneous_broken_vacuum_is_stationary_for_flat_curved_equation():
    potential = QuarticAgencityPotential(lambda_=2.0, mu=0.5)
    phi = np.full(4, vacuum_state(2.0, 0.5, theta=0.7), dtype=complex)
    residual = curved_field_residual(
        np.zeros_like(phi), phi, potential, scalar_curvature=0.0, xi=0.0
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-14)


def test_global_u1_invariance_of_matter_lagrangian_and_modulus():
    potential = QuarticAgencityPotential(lambda_=1.3, mu=0.9)
    phi = np.array([0.3 + 0.2j, -0.5 + 0.1j])
    derivatives = np.array(
        [
            [0.2 + 0.1j, 0.0, 0.3j, -0.1],
            [0.0, 0.4 - 0.2j, -0.3, 0.1j],
        ]
    )
    alpha = 1.234
    phase = np.exp(1j * alpha)
    before = matter_lagrangian_density(
        phi, derivatives, minkowski_inverse_metric(), potential
    )
    after = matter_lagrangian_density(
        phi * phase,
        derivatives * phase,
        minkowski_inverse_metric(),
        potential,
    )
    np.testing.assert_allclose(np.abs(phi * phase), np.abs(phi))
    np.testing.assert_allclose(potential.value(phi * phase), potential.value(phi))
    np.testing.assert_allclose(after, before)


def test_minimal_stress_energy_is_real_symmetric_and_u1_invariant():
    potential = QuarticAgencityPotential(lambda_=0.7, mu=1.1)
    phi = np.array([0.4 + 0.2j, -0.1 + 0.3j])
    derivatives = np.array(
        [
            [0.5 + 0.1j, 0.2j, -0.3, 0.1 + 0.2j],
            [-0.2j, 0.4 + 0.1j, 0.0, -0.25],
        ]
    )
    metric = minkowski_metric()
    inverse = minkowski_inverse_metric()
    tensor = stress_energy_tensor(phi, derivatives, metric, inverse, potential, xi=0.0)
    np.testing.assert_allclose(tensor, np.swapaxes(tensor, -1, -2))
    assert np.isrealobj(tensor)

    phase = np.exp(0.77j)
    rotated = stress_energy_tensor(
        phi * phase,
        derivatives * phase,
        metric,
        inverse,
        potential,
        xi=0.0,
    )
    np.testing.assert_allclose(rotated, tensor)


def test_nonminimal_stress_energy_is_explicitly_unsupported():
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    with pytest.raises(NotImplementedError, match="nonminimal"):
        stress_energy_tensor(
            0.0,
            np.zeros(4),
            minkowski_metric(),
            minkowski_inverse_metric(),
            potential,
            xi=1.0 / 6.0,
        )


def test_einstein_residual_exact_constructed_zero():
    stress = np.arange(16.0).reshape(4, 4)
    stress = 0.5 * (stress + stress.T)
    gravitational_constant = 0.25
    einstein = 8.0 * np.pi * gravitational_constant * stress
    residual = einstein_equation_residual(einstein, stress, gravitational_constant)
    np.testing.assert_allclose(residual, 0.0)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_gravitational_constants_are_rejected(value):
    with pytest.raises(ValueError, match="gravitational_constant"):
        einstein_equation_residual(np.zeros((4, 4)), np.zeros((4, 4)), value)
    with pytest.raises(ValueError, match="gravitational_constant"):
        einstein_hilbert_density(1.0, 0.0, gravitational_constant=value)


@pytest.mark.parametrize("xi", [np.nan, np.inf, -np.inf])
def test_nonfinite_xi_is_rejected(xi):
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    with pytest.raises(ValueError, match="xi"):
        nonminimal_coupling_density(1.0, 1.0, 0.0, xi=xi)
    with pytest.raises(ValueError, match="xi"):
        curved_field_residual(0.0, 0.0, potential, 0.0, xi=xi)


def test_tensor_shape_validation_rejects_ambiguous_inputs():
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    phi = np.ones(2)
    derivatives = np.zeros((2, 4))
    with pytest.raises(ValueError, match="inverse_metric"):
        matter_lagrangian_density(phi, derivatives, np.eye(3), potential)
    with pytest.raises(ValueError, match="identical shapes"):
        einstein_equation_residual(np.zeros((4, 4)), np.zeros((2, 4, 4)), 1.0)


def test_gravity_flat_operator_and_chapter16_dynamics_signature_difference_are_protected():
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    x = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    grid = UniformRectilinearGrid(axes=(x,))
    phi = 0.2 * np.cos(x)
    spatial_laplacian = laplacian(phi, grid)

    # Existing Chapter-16 dynamics uses (+,-,-,-):
    # phi_tt - laplacian(phi) + potential.gradient(phi) = 0.
    phi_tt = klein_gordon_acceleration(phi, grid, potential)
    chapter16_residual = phi_tt - spatial_laplacian + potential.gradient(phi)
    np.testing.assert_allclose(chapter16_residual, 0.0, atol=1e-14)

    # Chapter 19 explicitly switches to (-,+,+,+) while retaining the source
    # equation box(phi) + potential.gradient(phi) = 0.  We implement that text
    # literally rather than silently changing either chapter.
    gravity_box = minkowski_box(phi_tt, spatial_laplacian)
    gravity_residual = curved_field_residual(
        gravity_box, phi, potential, scalar_curvature=0.0, xi=0.0
    )
    expected_gravity = -phi_tt + spatial_laplacian + potential.gradient(phi)
    np.testing.assert_allclose(gravity_residual, expected_gravity)
    assert not np.allclose(gravity_residual, chapter16_residual)


def test_gravity_package_intentionally_contains_no_solver_or_dynamics_dependency():
    assert not any(name.startswith("simulate_") for name in gravity.__all__)
    for module in (gravity.action, gravity.geometry, gravity.stress_energy, gravity.equations):
        assert "agencitylab.fields.dynamics" not in inspect.getsource(module)
