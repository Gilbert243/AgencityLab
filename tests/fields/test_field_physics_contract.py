"""Tests for the isolated research field-physics contract."""

from __future__ import annotations

import numpy as np
import pytest

from agencitylab.fields.local_field import compute_agencity_field
from agencitylab.fields.physics import (
    SCIENTIFIC_STATUS,
    QuarticAgencityPotential,
    beta_to_phi,
    dimensionless_benchmark,
    field_energy_density,
    gradient_energy_density,
    kinetic_energy_density,
    phi_from_observable_field,
    potential_energy_density,
    total_field_energy,
    vacuum_amplitude,
    vacuum_state,
)
from agencitylab.models.field_extensions import ParameterSource
from agencitylab.scientific_status import ScientificStatus


def test_physics_status_is_research():
    assert SCIENTIFIC_STATUS is ScientificStatus.RESEARCH


@pytest.mark.parametrize(
    "beta",
    [
        np.array([[1.0, -2.0], [0.5, 3.0]]),
        np.array([[1.0 + 2.0j, -2.0j], [0.5 - 0.25j, 3.0 + 1.0j]]),
    ],
)
def test_beta_to_phi_real_and_complex(beta):
    result = beta_to_phi(beta, 4.0, 0.25, time_axis=0)
    np.testing.assert_allclose(result, beta)
    assert np.iscomplexobj(result) == np.iscomplexobj(beta)


def test_beta_to_phi_scalar_power_and_tau_exact_formula():
    beta = np.array([[1.0 + 1.0j, 2.0], [3.0, 4.0 - 1.0j]])
    phi = beta_to_phi(beta, 9.0, 4.0)
    np.testing.assert_allclose(phi, 6.0 * beta)


def test_beta_to_phi_spatial_power_and_tau_time_axis_zero():
    beta = np.arange(24.0).reshape(4, 2, 3).astype(complex) + 1j
    power = np.array([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]])
    tau = np.array([[1.0, 4.0, 1.0], [4.0, 1.0, 4.0]])
    phi = beta_to_phi(beta, power, tau, time_axis=0)
    expected = np.sqrt(power * tau)[None, :, :] * beta
    np.testing.assert_allclose(phi, expected)
    assert np.all(phi[:, 0, 0] == 0.0)


def test_beta_to_phi_spacetime_power_time_axis_zero():
    beta = np.ones((3, 2), dtype=complex) * (1.0 + 2.0j)
    power = np.array([[0.0, 1.0], [4.0, 9.0], [16.0, 25.0]])
    phi = beta_to_phi(beta, power, 4.0, time_axis=0)
    np.testing.assert_allclose(phi, np.sqrt(power * 4.0) * beta)


def test_beta_to_phi_time_axis_minus_one():
    beta = np.arange(12.0).reshape(2, 2, 3).astype(complex) + 1j
    power = np.array([[1.0, 4.0], [9.0, 16.0]])
    tau = np.array([[4.0, 1.0], [4.0, 1.0]])
    phi = beta_to_phi(beta, power, tau, time_axis=-1)
    expected = beta * np.sqrt(power * tau)[:, :, None]
    np.testing.assert_allclose(phi, expected)


def test_beta_to_phi_spacetime_power_time_axis_minus_one():
    beta = np.ones((2, 3), dtype=complex)
    power = np.arange(6.0).reshape(2, 3)
    phi = beta_to_phi(beta, power, 2.0, time_axis=-1)
    np.testing.assert_allclose(phi, np.sqrt(2.0 * power) * beta)


@pytest.mark.parametrize("power", [-1.0, np.array([1.0, -1.0])])
def test_beta_to_phi_rejects_negative_power(power):
    beta = np.ones((3, 2))
    with pytest.raises(ValueError, match="P_c"):
        beta_to_phi(beta, power, 1.0)


@pytest.mark.parametrize("tau", [0.0, -1.0, np.array([1.0, 0.0])])
def test_beta_to_phi_rejects_nonpositive_tau(tau):
    beta = np.ones((3, 2))
    with pytest.raises(ValueError, match="tau"):
        beta_to_phi(beta, 1.0, tau)


@pytest.mark.parametrize(
    ("name", "power", "tau"),
    [
        ("P_c", np.nan, 1.0),
        ("P_c", np.inf, 1.0),
        ("tau", 1.0, np.nan),
        ("tau", 1.0, np.inf),
    ],
)
def test_beta_to_phi_rejects_nonfinite_parameters(name, power, tau):
    with pytest.raises(ValueError, match=name):
        beta_to_phi(np.ones((3, 2)), power, tau)


def test_beta_to_phi_rejects_ambiguous_numpy_broadcast_shapes():
    beta = np.ones((4, 2, 3))
    with pytest.raises(ValueError, match="exact spatial shape"):
        beta_to_phi(beta, np.ones(3), 1.0)
    with pytest.raises(ValueError, match="exact spatial shape"):
        beta_to_phi(beta, 1.0, np.ones(3))


def test_beta_to_phi_rejects_spacetime_tau():
    beta = np.ones((4, 2, 3))
    with pytest.raises(ValueError, match="spacetime tau"):
        beta_to_phi(beta, 1.0, np.ones_like(beta))


def test_observable_field_bridge_is_explicit_exact_and_nonmutating():
    t = np.linspace(0.0, 6.0, 41)
    u = np.stack([np.sin(t), 0.5 * np.cos(t)], axis=1)
    result = compute_agencity_field(
        u,
        t,
        A_ref=np.array([1.0, 2.0]),
        tau=np.array([1.0, 2.0]),
        w=np.array([1.0, 2.0]),
        P_c=np.array([0.0, 4.0]),
        time_axis=0,
    )
    beta_before = result.beta.copy()
    power_before = result.P_c.copy()
    phi = phi_from_observable_field(result)
    expected = np.sqrt(result.P_c * result.tau[None, :]) * result.beta
    np.testing.assert_allclose(phi, expected)
    np.testing.assert_array_equal(result.beta, beta_before)
    np.testing.assert_array_equal(result.P_c, power_before)
    assert np.all(phi[:, 0] == 0.0)


def test_quartic_potential_zero_real_complex_and_exact_value():
    potential = QuarticAgencityPotential(lambda_=2.0, mu=3.0)
    assert potential.value(0.0) == pytest.approx(0.0)
    phi = np.array([2.0, 1.0 + 2.0j])
    modulus_squared = np.abs(phi) ** 2
    expected = -modulus_squared + 0.75 * modulus_squared**2
    np.testing.assert_allclose(potential.value(phi), expected)


def test_quartic_potential_u1_invariance():
    potential = QuarticAgencityPotential(lambda_=1.5, mu=0.75)
    phi = np.array([1.0 + 2.0j, -0.25 + 0.5j])
    for theta in (0.1, 1.7, -2.4):
        np.testing.assert_allclose(
            potential.value(phi * np.exp(1j * theta)), potential.value(phi)
        )


def test_quartic_potential_gradient_exact_for_real_and_complex_fields():
    potential = QuarticAgencityPotential(lambda_=-0.5, mu=2.0)
    phi = np.array([2.0, 1.0 + 2.0j])
    expected = 0.5 * phi + 2.0 * np.abs(phi) ** 2 * phi
    np.testing.assert_allclose(potential.gradient(phi), expected)


@pytest.mark.parametrize("mu", [0.0, -1.0, np.nan, np.inf])
def test_quartic_potential_rejects_invalid_mu(mu):
    with pytest.raises(ValueError, match="mu"):
        QuarticAgencityPotential(lambda_=1.0, mu=mu)


@pytest.mark.parametrize("lambda_", [np.nan, np.inf, -np.inf])
def test_quartic_potential_rejects_nonfinite_lambda(lambda_):
    with pytest.raises(ValueError, match="lambda"):
        QuarticAgencityPotential(lambda_=lambda_, mu=1.0)


def test_wirtinger_gradient_matches_real_component_finite_differences():
    potential = QuarticAgencityPotential(lambda_=1.3, mu=0.8)
    phi = 0.7 - 0.4j
    gradient = potential.gradient(phi)
    h = 1e-6
    d_v_dx = (potential.value(phi + h) - potential.value(phi - h)) / (2.0 * h)
    d_v_dy = (potential.value(phi + 1j * h) - potential.value(phi - 1j * h)) / (2.0 * h)
    # For real V, dV/d(phi*) = 1/2 (dV/dx + i dV/dy).
    assert d_v_dx == pytest.approx(2.0 * np.real(gradient), rel=1e-7, abs=1e-8)
    assert d_v_dy == pytest.approx(2.0 * np.imag(gradient), rel=1e-7, abs=1e-8)


def test_vacuum_amplitude_and_gradient_stationarity():
    potential = QuarticAgencityPotential(lambda_=2.0, mu=0.5)
    v = vacuum_amplitude(2.0, 0.5)
    assert v == pytest.approx(2.0)
    for theta in (0.0, 0.4, 1.9, -2.2):
        phi = vacuum_state(2.0, 0.5, theta=theta)
        assert abs(phi) == pytest.approx(v)
        assert potential.gradient(phi) == pytest.approx(0.0j, abs=1e-12)


def test_vacuum_u1_phases_have_equal_energy():
    potential = QuarticAgencityPotential(lambda_=3.0, mu=2.0)
    energies = [potential.value(vacuum_state(3.0, 2.0, theta=theta)) for theta in (0.0, 1.0, 2.5)]
    np.testing.assert_allclose(energies, energies[0])


@pytest.mark.parametrize("lambda_", [0.0, -1.0])
def test_nonpositive_lambda_has_no_false_nonzero_vacuum(lambda_):
    with pytest.raises(ValueError, match="lambda_ > 0"):
        vacuum_amplitude(lambda_, 1.0)
    with pytest.raises(ValueError, match="lambda_ > 0"):
        vacuum_state(lambda_, 1.0, theta=0.0)


def test_vacuum_state_requires_explicit_phase():
    with pytest.raises(TypeError):
        vacuum_state(1.0, 1.0)


def test_kinetic_and_gradient_energy_density_are_real_for_complex_field():
    phi_dot = np.array([1.0 + 2.0j, -3.0j])
    kinetic = kinetic_energy_density(phi_dot)
    gradient = gradient_energy_density(np.array([4.0, 9.0]))
    np.testing.assert_allclose(kinetic, np.array([2.5, 4.5]))
    np.testing.assert_allclose(gradient, np.array([2.0, 4.5]))
    assert np.isrealobj(kinetic)
    assert np.isrealobj(gradient)


def test_gradient_energy_rejects_negative_or_significantly_complex_norms():
    with pytest.raises(ValueError, match="non-negative"):
        gradient_energy_density(np.array([1.0, -0.1]))
    with pytest.raises(ValueError, match="theoretically real"):
        gradient_energy_density(np.array([1.0 + 1e-3j]))


def test_gradient_energy_accepts_roundoff_level_imaginary_part():
    density = gradient_energy_density(np.array([4.0 + 1e-15j]))
    np.testing.assert_allclose(density, np.array([2.0]))
    assert np.isrealobj(density)


def test_field_energy_density_is_exact_component_sum_and_real():
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    phi = np.array([1.0 + 1.0j, 0.5j])
    phi_dot = np.array([0.0j, 2.0 - 1.0j])
    grad2 = np.array([0.0, 3.0])
    kinetic = kinetic_energy_density(phi_dot)
    gradient = gradient_energy_density(grad2)
    potential_density = potential_energy_density(phi, potential)
    total = field_energy_density(phi, phi_dot, grad2, potential)
    np.testing.assert_allclose(total, kinetic + gradient + potential_density)
    assert np.isrealobj(total)
    assert np.all(np.isfinite(total))


def test_total_field_energy_uses_explicit_scalar_volume_element():
    density = np.array([1.0, 2.0, 3.0])
    assert total_field_energy(density, volume_element=0.5) == pytest.approx(3.0)


def test_total_field_energy_uses_explicit_array_weights():
    density = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([[1.0, 0.5], [0.25, 2.0]])
    assert total_field_energy(density, volume_element=weights) == pytest.approx(
        np.sum(density * weights)
    )


def test_total_field_energy_rejects_implicit_broadcasting():
    with pytest.raises(ValueError, match="exact density shape"):
        total_field_energy(np.ones((2, 3)), volume_element=np.ones(3))


def test_dimensionless_benchmark_is_explicit_and_provenanced():
    potential = dimensionless_benchmark()
    assert potential.lambda_ == pytest.approx(1.0)
    assert potential.mu == pytest.approx(1.0)
    assert potential.units_convention == "dimensionless"
    assert potential.scientific_status is ScientificStatus.RESEARCH
    assert potential.lambda_provenance is not None
    assert potential.mu_provenance is not None
    assert potential.lambda_provenance.source is ParameterSource.DIMENSIONLESS_BENCHMARK
    assert "not a universal physical parameter" in potential.lambda_provenance.note


def test_potential_has_no_hidden_physical_defaults():
    with pytest.raises(TypeError):
        QuarticAgencityPotential()
