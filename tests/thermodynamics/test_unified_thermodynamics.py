from __future__ import annotations

import inspect

import numpy as np
import pytest

import agencitylab
from agencitylab.fields.dynamics import simulate_dissipative_klein_gordon
from agencitylab.fields.numerics import PeriodicBoundary, UniformRectilinearGrid
from agencitylab.fields.physics import QuarticAgencityPotential
from agencitylab.models.field_extensions import (
    DynamicalAgencityFieldSolution,
    ParameterSource,
)
from agencitylab.scientific_status import ScientificStatus
from agencitylab.thermodynamics.dissipation import (
    SCIENTIFIC_STATUS as DISSIPATION_STATUS,
    dissipation_density,
    entropy_production_density,
    total_dissipated_power,
    total_entropy_production,
)
from agencitylab.thermodynamics.effective_temperature import (
    SCIENTIFIC_STATUS as TEMPERATURE_STATUS,
    temperature_dependent_lambda,
)
from agencitylab.thermodynamics.energy_balance import energy_balance_residual
from agencitylab.thermodynamics.entropy import (
    SCIENTIFIC_STATUS as ENTROPY_STATUS,
    contrast_agencial_entropy,
    field_agencial_entropy,
)
from agencitylab.thermodynamics.landauer import (
    landauer_agencity_power,
    landauer_characteristic_power,
    structural_information_rate,
)
from agencitylab.thermodynamics.laws import (
    SCIENTIFIC_STATUS as LAWS_STATUS,
    PhaseLawFit,
    modulus_law_margin,
    modulus_law_satisfied,
    phase_law_prediction,
    phase_law_residual,
    phi_imaginary_component,
    second_law_residual,
    thermal_reference_phase_fit,
)


def test_thermodynamics_remains_research_when_selected_apis_are_globally_exported():
    assert agencitylab.__version__ == "1.1.3"
    assert DISSIPATION_STATUS is ScientificStatus.RESEARCH
    assert TEMPERATURE_STATUS is ScientificStatus.RESEARCH
    assert ENTROPY_STATUS is ScientificStatus.RESEARCH
    assert LAWS_STATUS is ScientificStatus.RESEARCH
    assert agencitylab.field_agencial_entropy is field_agencial_entropy
    assert agencitylab.phase_law_prediction is phase_law_prediction
    assert agencitylab.dissipation_density is dissipation_density
    assert agencitylab.temperature_dependent_lambda is temperature_dependent_lambda


def test_zero_velocity_gives_exact_zero_dissipation_and_entropy_production():
    phi_dot = np.zeros((4, 3), dtype=complex)
    q_dot = dissipation_density(phi_dot, gamma=2.5)
    sigma = entropy_production_density(phi_dot, gamma=2.5, t_eff=300.0)

    np.testing.assert_array_equal(q_dot, np.zeros((4, 3)))
    np.testing.assert_array_equal(sigma, np.zeros((4, 3)))


def test_zero_gamma_gives_exact_zero_without_regularisation():
    phi_dot = np.array([1.0 + 2.0j, -3.0j, 4.0])
    result = dissipation_density(phi_dot, gamma=0.0)
    np.testing.assert_array_equal(result, np.zeros(3))


@pytest.mark.parametrize("gamma", [-1.0, np.nan, np.inf, -np.inf])
def test_dissipation_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        dissipation_density(np.ones(4), gamma=gamma)


@pytest.mark.parametrize("complex_field", [False, True])
def test_positive_gamma_dissipation_is_nonnegative_for_real_and_complex_fields(complex_field):
    phi_dot = np.arange(12, dtype=float).reshape(3, 4) / 10.0
    if complex_field:
        phi_dot = phi_dot + 0.3j * np.flip(phi_dot, axis=1)

    result = dissipation_density(phi_dot, gamma=0.7)

    assert result.shape == phi_dot.shape
    assert np.isrealobj(result)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(result, 0.7 * np.abs(phi_dot) ** 2)


@pytest.mark.parametrize("t_eff", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_entropy_production_rejects_nonpositive_or_nonfinite_temperature(t_eff):
    with pytest.raises(ValueError):
        entropy_production_density(np.ones(4), gamma=1.0, t_eff=t_eff)


def test_entropy_production_accepts_positive_spatial_temperature_field():
    phi_dot = np.array([[1.0, 2.0], [3.0, 4.0]])
    t_eff = np.array([[2.0, 4.0], [3.0, 8.0]])
    result = entropy_production_density(phi_dot, gamma=0.5, t_eff=t_eff)
    np.testing.assert_allclose(result, 0.5 * phi_dot**2 / t_eff)


def test_total_dissipation_and_entropy_production_reuse_nd_grid_quadrature():
    grid = UniformRectilinearGrid(shape=(3, 4), spacings=(0.5, 0.25))
    phi_dot = np.full(grid.shape, 2.0 + 1.0j)
    q_density = 2.0 * np.abs(phi_dot) ** 2
    expected_power = float(np.sum(q_density) * grid.cell_volume)

    power = total_dissipated_power(phi_dot, gamma=2.0, grid=grid)
    production = total_entropy_production(
        phi_dot,
        gamma=2.0,
        t_eff=5.0,
        grid=grid,
    )

    assert power == pytest.approx(expected_power)
    assert production == pytest.approx(expected_power / 5.0)


def test_temperature_dependent_lambda_crosses_zero_at_critical_temperature():
    a = 2.0
    t_c = 10.0
    assert temperature_dependent_lambda(12.0, a, t_c) == pytest.approx(-4.0)
    assert temperature_dependent_lambda(10.0, a, t_c) == pytest.approx(0.0)
    assert temperature_dependent_lambda(8.0, a, t_c) == pytest.approx(4.0)


def test_temperature_dependent_lambda_does_not_hide_a_sign_constraint():
    values = temperature_dependent_lambda(
        np.array([8.0, 10.0, 12.0]),
        a=-2.0,
        t_c=10.0,
    )
    np.testing.assert_allclose(values, [-4.0, 0.0, 4.0])


def test_field_agencial_entropy_zero_uniform_and_volume_are_exactly_controlled():
    grid = UniformRectilinearGrid(shape=(4, 5), spacings=(0.5, 0.25))
    zero = np.zeros(grid.shape)
    uniform = np.full(grid.shape, 3.0)
    a = 2.0

    assert field_agencial_entropy(zero, a=a, grid=grid) == 0.0
    expected = 0.5 * a * np.sum(np.abs(uniform) ** 2) * grid.cell_volume
    assert field_agencial_entropy(uniform, a=a, grid=grid) == pytest.approx(expected)


def test_field_agencial_entropy_is_u1_invariant_and_scales_with_modulus_squared():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    x = grid.axes[0]
    phi = (0.3 + 0.1 * np.sin(x)) * np.exp(1j * x)
    original = phi.copy()
    base = field_agencial_entropy(phi, a=1.7, grid=grid)
    rotated = field_agencial_entropy(phi * np.exp(0.73j), a=1.7, grid=grid)
    scaled = field_agencial_entropy(4.0 * phi, a=1.7, grid=grid)

    assert rotated == pytest.approx(base)
    assert scaled == pytest.approx(16.0 * base)
    np.testing.assert_array_equal(phi, original)


def test_contrast_agencial_entropy_reproduces_volume1_h26():
    j = np.array([-0.2, 0.0, 0.4])
    j_max = 1.0
    k_b = 1.380649e-23
    expected = -k_b * np.log(1.0 - np.abs(j) / j_max)
    result = contrast_agencial_entropy(j, j_max=j_max, k_b=k_b)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("j,j_max", [(1.0, 1.0), (1.1, 1.0), (0.1, 0.0), (0.1, -1.0)])
def test_contrast_agencial_entropy_enforces_real_logarithm_domain(j, j_max):
    with pytest.raises(ValueError):
        contrast_agencial_entropy(j, j_max=j_max, k_b=1.0)


def test_field_and_contrast_agencial_entropies_are_explicitly_distinct():
    assert field_agencial_entropy is not contrast_agencial_entropy
    assert "phi" in inspect.signature(field_agencial_entropy).parameters
    assert "J" in inspect.signature(contrast_agencial_entropy).parameters


def test_energy_balance_residual_has_exact_constructed_zero():
    phi_dot = np.array([2.0, 3.0])
    gamma = 0.5
    dissipated = gamma * np.abs(phi_dot) ** 2
    dH_dt = np.array([-3.0, -5.5])
    div_j_e = np.array([1.0, 1.0])

    result = energy_balance_residual(dH_dt, div_j_e, phi_dot, gamma)
    np.testing.assert_array_equal(result, dH_dt + div_j_e + dissipated)
    np.testing.assert_array_equal(result, np.zeros(2))


def test_second_law_residual_is_diagnostic_and_not_clipped():
    assert second_law_residual(0.4, 0.6, 1.0) == pytest.approx(0.0)
    assert second_law_residual(0.2, 0.3, 1.0) == pytest.approx(-0.5)


def test_modulus_law_satisfied_equality_violation_and_signed_entropy_rate():
    assert modulus_law_margin(10.0 + 0j, 4.0, 2.0, 2.0) == pytest.approx(2.0)
    assert modulus_law_satisfied(10.0, 4.0, 2.0, 2.0) is True
    assert modulus_law_margin(8.0, 4.0, 2.0, 2.0) == pytest.approx(0.0)
    assert modulus_law_satisfied(8.0, 4.0, 2.0, 2.0) is True
    assert modulus_law_margin(7.0, 4.0, 2.0, 2.0) == pytest.approx(-1.0)
    assert modulus_law_satisfied(7.0, 4.0, 2.0, 2.0) is False
    assert modulus_law_margin(5.0, 4.0, 2.0, -1.0) == pytest.approx(3.0)


def test_phase_law_matches_controlled_analytic_value_with_explicit_fit():
    # Ratio = 100 / (2 * 5) = 10, so log10(ratio) = 1.
    value = phase_law_prediction(
        100.0,
        2.0,
        5.0,
        alpha=2.0,
        beta_fit=-3.0,
    )
    assert value == pytest.approx(-1.0)
    residual = phase_law_residual(
        -0.7,
        100.0,
        2.0,
        5.0,
        alpha=2.0,
        beta_fit=-3.0,
    )
    assert residual == pytest.approx(0.3)


def test_phase_law_accepts_explicit_user_fit_object():
    fit = PhaseLawFit(alpha=0.5, beta_fit=1.0)
    value = phase_law_prediction(100.0, 2.0, 5.0, fit=fit)
    assert value == pytest.approx(1.5)
    assert fit.reference_kind == "user_supplied_fit"
    assert fit.scientific_status is ScientificStatus.RESEARCH


def test_named_phase_fit_is_empirical_reference_with_explicit_provenance():
    fit = thermal_reference_phase_fit()
    data = fit.to_dict()

    assert fit.alpha == pytest.approx(0.82)
    assert fit.beta_fit == pytest.approx(-1.50)
    assert fit.r_squared == pytest.approx(0.87)
    assert fit.reference_kind == "empirical_reference"
    assert fit.scientific_status is ScientificStatus.RESEARCH
    assert data["scientific_status"] == "research"
    assert data["provenance"]["alpha"]["source"] == "source_document_reference"
    assert fit.provenance["alpha"].source is ParameterSource.SOURCE_DOCUMENT_REFERENCE


@pytest.mark.parametrize(
    "p_diss,t_amb,sdot_int",
    [
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
    ],
)
def test_phase_law_rejects_undefined_or_nonpositive_ratio(p_diss, t_amb, sdot_int):
    with pytest.raises(ValueError):
        phase_law_prediction(
            p_diss,
            t_amb,
            sdot_int,
            alpha=1.0,
            beta_fit=0.0,
        )


def test_phase_law_api_cannot_accidentally_consume_canonical_organisation_O():
    prediction_parameters = inspect.signature(phase_law_prediction).parameters
    residual_parameters = inspect.signature(phase_law_residual).parameters

    assert "O" not in prediction_parameters
    assert "O" not in residual_parameters
    assert "phase_component" in residual_parameters

    phi = np.array([1.0 + 2.0j, 3.0 - 4.0j])
    np.testing.assert_array_equal(phi_imaginary_component(phi), [2.0, -4.0])


def test_conditional_landauer_relations_match_volume1_h23_h25_without_redefining_b():
    k_b = 2.0
    t_eff = 3.0
    tau = 4.0
    beta = np.array([3.0 + 4.0j, 0.0])

    p_c = landauer_characteristic_power(k_b, t_eff, tau)
    information_rate = structural_information_rate(beta, tau)
    power = landauer_agencity_power(k_b, t_eff, information_rate)

    assert p_c == pytest.approx(1.5)
    np.testing.assert_allclose(information_rate, [1.25, 0.0])
    np.testing.assert_allclose(power, [7.5, 0.0])
    np.testing.assert_allclose(power, p_c * np.abs(beta))


def test_dissipative_solver_states_are_consistent_with_gamma_phi_dot_squared():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi0 = np.full(grid.shape, 0.1)
    phi_dot0 = np.full(grid.shape, 0.2)
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    gamma = 0.4

    solution = simulate_dissipative_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        gamma=gamma,
        dt=0.001,
        n_steps=2,
        boundary=PeriodicBoundary(),
    )

    assert isinstance(solution, DynamicalAgencityFieldSolution)
    assert solution.phi_dot is not None
    assert solution.parameters["Gamma"] == gamma
    assert solution.parameter_provenance["Gamma"].source is ParameterSource.USER_SUPPLIED
    assert solution.scientific_status is ScientificStatus.RESEARCH

    q_dot = dissipation_density(solution.phi_dot[1], gamma)
    np.testing.assert_allclose(q_dot, gamma * np.abs(solution.phi_dot[1]) ** 2)


def test_public_evaluators_do_not_mutate_input_arrays():
    phi_dot = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    phi_dot_before = phi_dot.copy()
    b = np.array([2.0 + 3.0j, 4.0 - 1.0j])
    b_before = b.copy()
    sdot = np.array([0.5, -0.25])
    sdot_before = sdot.copy()

    dissipation_density(phi_dot, 0.3)
    entropy_production_density(phi_dot, 0.3, 2.0)
    modulus_law_margin(b, 1.0, 2.0, sdot)

    np.testing.assert_array_equal(phi_dot, phi_dot_before)
    np.testing.assert_array_equal(b, b_before)
    np.testing.assert_array_equal(sdot, sdot_before)
