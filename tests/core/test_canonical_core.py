import importlib

import numpy as np
import pytest

from agencitylab import compute_agencity
from agencitylab.core.activation import activation, reduced_coordinate
from agencitylab.core.activity import activity
from agencitylab.core.agencity import agencity
from agencitylab.core.beta import compute_beta
from agencitylab.core.contrast import compute_contrast
from agencitylab.core.crm import causal_moving_correlation
from agencitylab.core.intensity import (
    compute_dynamic_intensity,
    compute_structural_intensity,
)
from agencitylab.core.memory import memory
from agencitylab.core.normalization import normalize_signal
from agencitylab.core.organization import organization
from agencitylab.core.orientation import compute_orientation
from agencitylab.core.power import characteristic_power
from agencitylab.core.tau import characteristic_time


def test_normalization_is_exact_physical_ratio():
    u = np.array([-4.0, 0.0, 8.0])
    u_star, ref = normalize_signal(u, A_ref=2.0)
    np.testing.assert_array_equal(u_star, np.array([-2.0, 0.0, 4.0]))
    assert ref == 2.0


def test_unknown_reference_amplitude_is_not_invented():
    with pytest.raises(ValueError, match="A_ref"):
        normalize_signal(np.array([1.0, 2.0, 3.0]))


def test_reduced_time_activation_and_activity_are_analytic_for_quadratic():
    t = np.linspace(-2.0, 2.0, 21)
    tau = 2.0
    t_star = reduced_coordinate(t, tau)
    u_star = t_star**2
    X_star = activation(u_star, t_star)
    A_star = activity(X_star, t_star)
    np.testing.assert_allclose(t_star, t / tau, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(X_star, 2.0 * t_star, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(A_star, 2.0, rtol=1e-12, atol=1e-12)


def test_exact_constant_activation_is_zero_without_epsilon_threshold():
    t_star = np.linspace(0.0, 4.0, 41)
    u_star = np.ones_like(t_star)
    np.testing.assert_array_equal(activation(u_star, t_star), 0.0)


def test_crm_repeating_and_inverted_blocks():
    axis = np.arange(8.0)
    repeated = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    inverted = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    assert causal_moving_correlation(repeated, 4.0, axis=axis)[-1] == pytest.approx(1.0)
    assert causal_moving_correlation(inverted, 4.0, axis=axis)[-1] == pytest.approx(-1.0)


def test_crm_constant_variance_convention_is_zero():
    axis = np.arange(8.0)
    signal = np.ones(8)
    assert causal_moving_correlation(signal, 4.0, axis=axis)[-1] == 0.0


def test_crm_has_no_epsilon_physics_for_small_nonzero_signal():
    axis = np.arange(8.0)
    block = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-30
    signal = np.concatenate([block, block])
    assert causal_moving_correlation(signal, 4.0, axis=axis)[-1] == pytest.approx(1.0)


def test_cross_crm_and_canonical_memory_organization():
    axis = np.arange(8.0)
    repeated = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    X_star = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    M = memory(repeated, 4.0, axis=axis)
    O = organization(repeated, X_star, 4.0, axis=axis)
    assert M[-1] == pytest.approx(1.0)
    assert O[-1] == pytest.approx(1.0)
    assert M[-1] != pytest.approx(np.tanh(1.0))


def test_dynamic_and_structural_intensities_match_definitions():
    X = np.array([3.0])
    A = np.array([4.0])
    M = np.array([3.0])
    O = np.array([4.0])
    D = compute_dynamic_intensity(X, A)
    S = compute_structural_intensity(M, O)
    np.testing.assert_allclose(D, np.sqrt(9.0 + 144.0))
    np.testing.assert_allclose(S, 5.0)


def test_contrast_is_exact_and_zero_when_intensities_match():
    D = np.array([0.0, 1.0, 4.0])
    S = np.array([1.0, 1.0, 2.0])
    expected = np.log((np.e + D) / (np.e + S))
    np.testing.assert_array_equal(compute_contrast(D, S), expected)
    assert compute_contrast(np.array([3.0]), np.array([3.0]))[0] == 0.0


def test_orientation_uses_exact_positive_S_even_far_below_eps():
    M = np.array([3.0, 1e-30, 0.0])
    O = np.array([4.0, 0.0, 0.0])
    U, S = compute_orientation(M, O, return_intensity=True)
    assert U[0] == pytest.approx(0.6 + 0.8j)
    assert S[1] == 1e-30
    assert U[1] == 1.0 + 0.0j
    assert U[2] == 0.0 + 0.0j


def test_beta_is_zero_exactly_when_structure_is_zero():
    D = np.array([5.0])
    M = np.array([0.0])
    O = np.array([0.0])
    S = np.array([0.0])
    J, U, beta = compute_beta(D, S, M, O)
    assert J[0] > 0.0
    assert U[0] == 0.0j
    assert beta[0] == 0.0j


def test_beta_modulus_equals_absolute_contrast_when_S_positive():
    M = np.array([0.6])
    O = np.array([0.8])
    S = np.array([1.0])
    D = np.array([2.0])
    J, _, beta = compute_beta(D, S, M, O)
    np.testing.assert_allclose(np.abs(beta), np.abs(J), rtol=1e-14, atol=0.0)


def test_flux_is_linear_in_characteristic_power():
    beta = np.array([1.0 + 2.0j, -0.5j])
    np.testing.assert_array_equal(agencity(beta, 3.0), 3.0 * beta)


def test_structural_parameters_have_no_arbitrary_auto_fallback():
    assert characteristic_time(tau=2.5) == 2.5
    assert characteristic_power(value=7.0) == 7.0
    with pytest.raises(ValueError, match="tau"):
        characteristic_time()
    with pytest.raises(ValueError, match="P_c"):
        characteristic_power()


def test_characteristic_power_from_documented_reference_energy_and_tau():
    assert characteristic_power(reference_energy=20.0, tau=4.0) == 5.0
    assert characteristic_power(reference_energy=20.0, inertia=2.0, tau=4.0) == 10.0


def test_canonical_pipeline_identities_and_window():
    t = np.linspace(0.0, 6.0, 61)
    u = np.sin(2.0 * np.pi * t)
    result = compute_agencity(u=u, xi=t, A_ref=2.0, tau=1.0, P_c=5.0)

    np.testing.assert_allclose(result.t_star, t)
    np.testing.assert_allclose(result.u_star, u / 2.0)
    np.testing.assert_allclose(result.D, np.hypot(result.X_star, result.A_star * result.X_star))
    np.testing.assert_allclose(result.S, np.hypot(result.M, result.O))
    np.testing.assert_allclose(result.J, np.log((np.e + result.D) / (np.e + result.S)))
    valid = result.S > 0.0
    np.testing.assert_allclose(np.abs(result.U[valid]), 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.beta[~valid], 0.0j)
    np.testing.assert_allclose(result.beta[valid], result.J[valid] * result.U[valid])
    np.testing.assert_allclose(result.b, 5.0 * result.beta)
    assert result.metadata.extra["memory_window"] == result.tau


def test_constant_signal_has_exact_postulated_null_state():
    t = np.linspace(0.0, 4.0, 41)
    result = compute_agencity(u=np.ones_like(t), xi=t, A_ref=1.0, tau=1.0, P_c=2.0)
    np.testing.assert_array_equal(result.X_star, 0.0)
    np.testing.assert_array_equal(result.A_star, 0.0)
    np.testing.assert_array_equal(result.M, 0.0)
    np.testing.assert_array_equal(result.O, 0.0)
    np.testing.assert_array_equal(result.D, 0.0)
    np.testing.assert_array_equal(result.S, 0.0)
    np.testing.assert_array_equal(result.J, 0.0)
    np.testing.assert_array_equal(result.U, 0.0j)
    np.testing.assert_array_equal(result.beta, 0.0j)
    np.testing.assert_array_equal(result.b, 0.0j)


def test_constant_signal_bypasses_derivative_and_crm_stages(monkeypatch):
    compute_module = importlib.import_module("agencitylab.api.compute")

    def unexpected(*args, **kwargs):
        raise AssertionError("null state must bypass derivative and CRM operators")

    for name in ("activation", "activity", "memory", "organization"):
        monkeypatch.setattr(compute_module, name, unexpected)

    t = np.linspace(0.0, 4.0, 41)
    result = compute_module.compute_agencity(
        u=np.ones_like(t), xi=t, A_ref=1.0, tau=1.0, P_c=2.0
    )
    np.testing.assert_array_equal(result.b, 0.0j)


def test_canonical_api_rejects_non_tau_memory_window():
    t = np.linspace(0.0, 4.0, 41)
    with pytest.raises(ValueError, match="w = tau"):
        compute_agencity(u=np.sin(t), xi=t, A_ref=1.0, tau=1.0, w=0.5, P_c=1.0)
