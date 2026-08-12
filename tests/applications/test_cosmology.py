from __future__ import annotations

import numpy as np
import pytest

import agencitylab
from agencitylab.applications.cosmology import (
    SCIENTIFIC_STATUS,
    acceleration_equation_residual,
    equation_of_state_parameter,
    field_acceleration,
    friedmann_constraint_residual,
    homogeneous_energy_density,
    homogeneous_pressure,
    hubble_derivative,
    initial_hubble_from_friedmann,
    simulate_flat_flrw,
)
from agencitylab.fields.physics import QuarticAgencityPotential, field_energy_density, vacuum_amplitude
from agencitylab.scientific_status import ScientificStatus


def test_cosmology_layer_is_speculative_and_selected_api_is_public() -> None:
    assert SCIENTIFIC_STATUS is ScientificStatus.SPECULATIVE
    assert agencitylab.simulate_flat_flrw is simulate_flat_flrw
    assert agencitylab.homogeneous_energy_density is homogeneous_energy_density
    assert agencitylab.friedmann_constraint_residual is friedmann_constraint_residual


def test_homogeneous_energy_reuses_shared_field_energy() -> None:
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=0.5)
    phi = np.array([0.2 + 0.1j, 0.5 - 0.3j])
    phi_dot = np.array([0.4 - 0.2j, -0.1 + 0.2j])
    expected = field_energy_density(phi, phi_dot, 0.0, potential)
    np.testing.assert_allclose(homogeneous_energy_density(phi, phi_dot, potential), expected)


def test_pressure_and_equation_of_state_for_kinetic_dominated_state() -> None:
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    rho = homogeneous_energy_density(0.0, 2.0, potential)
    pressure = homogeneous_pressure(0.0, 2.0, potential)
    assert float(rho) == pytest.approx(2.0)
    assert float(pressure) == pytest.approx(2.0)
    assert float(equation_of_state_parameter(rho, pressure)) == pytest.approx(1.0)


def test_equation_of_state_does_not_regularize_zero_density() -> None:
    with pytest.raises(ValueError, match="undefined"):
        equation_of_state_parameter(0.0, 0.0)


def test_field_acceleration_is_regular_at_phi_zero_without_eps() -> None:
    potential = QuarticAgencityPotential(lambda_=2.0, mu=1.0)
    assert complex(np.asarray(field_acceleration(0.0, 0.0, 1.0, potential)).item()) == 0.0j


def test_friedmann_and_acceleration_residual_build_exact_source_equations() -> None:
    gravitational = 0.2
    rho = 1.5
    pressure = 0.4
    hubble = np.sqrt((8.0 * np.pi * gravitational / 3.0) * rho)
    np.testing.assert_allclose(
        friedmann_constraint_residual(hubble, rho, gravitational_constant=gravitational),
        0.0,
        atol=1e-15,
    )
    hubble_dot = float(
        hubble_derivative(
            hubble,
            rho,
            pressure,
            gravitational_constant=gravitational,
        )
    )
    np.testing.assert_allclose(
        acceleration_equation_residual(
            hubble_dot,
            hubble,
            rho,
            pressure,
            gravitational_constant=gravitational,
        ),
        0.0,
        atol=1e-15,
    )


def test_initial_hubble_requires_explicit_branch() -> None:
    gravitational = 0.1
    rho = 2.0
    magnitude = np.sqrt((8.0 * np.pi * gravitational / 3.0) * rho)
    assert initial_hubble_from_friedmann(
        rho,
        gravitational_constant=gravitational,
        branch="expanding",
    ) == pytest.approx(magnitude)
    assert initial_hubble_from_friedmann(
        rho,
        gravitational_constant=gravitational,
        branch="contracting",
    ) == pytest.approx(-magnitude)
    with pytest.raises(ValueError, match="branch"):
        initial_hubble_from_friedmann(rho, gravitational_constant=gravitational, branch="auto")


def test_negative_quartic_vacuum_energy_is_not_silently_repaired() -> None:
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    vacuum = vacuum_amplitude(1.0, 1.0)
    rho = float(homogeneous_energy_density(vacuum, 0.0, potential))
    assert rho == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="negative rho"):
        initial_hubble_from_friedmann(
            rho,
            gravitational_constant=1.0,
            branch="expanding",
        )


def test_flat_flrw_solver_is_finite_and_tracks_friedmann_constraint() -> None:
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    solution = simulate_flat_flrw(
        phi0=0.2 + 0.1j,
        phi_dot0=0.0,
        scale_factor0=1.0,
        potential=potential,
        gravitational_constant=0.01,
        dt=0.005,
        steps=80,
        branch="expanding",
    )
    assert solution.scientific_status is ScientificStatus.SPECULATIVE
    assert solution.model == "flat_flrw_agencity_field"
    assert solution.times.shape == (81,)
    assert np.all(np.isfinite(solution.phi))
    assert np.all(np.isfinite(solution.hubble))
    assert np.all(solution.scale_factor > 0.0)
    assert solution.hubble[0] > 0.0
    np.testing.assert_allclose(solution.friedmann_residual[0], 0.0, atol=1e-15)
    assert np.max(np.abs(solution.friedmann_residual)) < 1e-8


def test_flat_flrw_solver_respects_global_u1_rotation_for_background_observables() -> None:
    potential = QuarticAgencityPotential(lambda_=-0.7, mu=0.8)
    kwargs = dict(
        scale_factor0=1.0,
        potential=potential,
        gravitational_constant=0.02,
        dt=0.004,
        steps=50,
        branch="expanding",
    )
    phi0 = 0.15 + 0.05j
    phi_dot0 = 0.02 - 0.01j
    angle = 0.73
    phase = np.exp(1j * angle)
    reference = simulate_flat_flrw(phi0=phi0, phi_dot0=phi_dot0, **kwargs)
    rotated = simulate_flat_flrw(phi0=phi0 * phase, phi_dot0=phi_dot0 * phase, **kwargs)
    np.testing.assert_allclose(rotated.rho, reference.rho, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(rotated.pressure, reference.pressure, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(rotated.hubble, reference.hubble, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(rotated.scale_factor, reference.scale_factor, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(rotated.phi, reference.phi * phase, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_factor0": 0.0, "gravitational_constant": 0.1, "dt": 0.01, "steps": 1},
        {"scale_factor0": 1.0, "gravitational_constant": 0.0, "dt": 0.01, "steps": 1},
        {"scale_factor0": 1.0, "gravitational_constant": 0.1, "dt": 0.0, "steps": 1},
        {"scale_factor0": 1.0, "gravitational_constant": 0.1, "dt": 0.01, "steps": 0},
    ],
)
def test_flat_flrw_solver_validates_numerical_inputs(kwargs) -> None:
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    with pytest.raises(ValueError):
        simulate_flat_flrw(
            phi0=0.1,
            phi_dot0=0.0,
            potential=potential,
            branch="expanding",
            **kwargs,
        )
