import numpy as np
import pytest

from agencitylab import compute_agencity
from agencitylab.analysis.multi_scale import agencity_multiscale, summarize_multiscale
from agencitylab.api.extensions import (
    compute_agencity_spectrum,
    compute_discrete_agencity,
    compute_multivariate_agencity,
    optimize_agencity_window,
    riemannian_extension_status,
)
from agencitylab.core.discrete import volume2_first_difference, volume2_second_difference


def _signal():
    xi = np.linspace(0.0, 24.0, 481)
    return xi, np.sin(xi)


def test_spectrum_default_w_equals_tau_and_matches_scalar_reference():
    xi, u = _signal()
    taus = np.array([1.0, 2.0, 3.0])
    spectrum = compute_agencity_spectrum(
        u,
        xi,
        taus,
        A_ref=1.0,
        P_c=2.0,
    )

    np.testing.assert_array_equal(spectrum["w"], taus)
    assert "fallback convention" in spectrum["window_mode"]
    assert spectrum["b"].shape == (3, xi.size)
    reference = compute_agencity(u=u, xi=xi, A_ref=1.0, tau=2.0, P_c=2.0)
    np.testing.assert_allclose(spectrum["b"][1], reference.b)
    np.testing.assert_allclose(spectrum["beta"][1], reference.beta)


def test_spectrum_can_hold_w_fixed_and_matches_scalar_theory_api():
    xi, u = _signal()
    spectrum = compute_agencity_spectrum(
        u,
        xi,
        [1.0, 2.0, 3.0],
        A_ref=1.0,
        P_c=1.0,
        windows=1.0,
    )
    np.testing.assert_array_equal(spectrum["w"], np.ones(3))
    assert spectrum["window_mode"] == "explicit independent w"

    reference = compute_agencity(u=u, xi=xi, A_ref=1.0, tau=2.0, w=1.0, P_c=1.0)
    np.testing.assert_allclose(spectrum["b"][1], reference.b)
    np.testing.assert_allclose(spectrum["beta"][1], reference.beta)
    assert reference.memory_window == 1.0


def test_analysis_multiscale_preserves_descriptive_compatibility_keys():
    xi, u = _signal()
    entries = agencity_multiscale(
        xi,
        u,
        [1.0, 2.0, 3.0],
        A_ref=1.0,
        P_c=2.0,
    )
    assert {"b_std", "beta_std", "P_c", "b_rms", "w"}.issubset(entries[0])

    summary = summarize_multiscale(entries)
    assert summary["tau_opt_beta"] == summary["tau_peak_beta"]
    assert summary["tau_opt_b"] == summary["tau_peak_b"]
    assert summary["status"].startswith("descriptive multiscale summary")


def test_window_optimisation_uses_explicit_discrete_candidates():
    xi, u = _signal()
    result = optimize_agencity_window(
        u,
        xi,
        tau=2.0,
        A_ref=1.0,
        P_c=1.0,
        candidates=[0.05, 0.5, 1.0, 2.0, 4.0],
    )

    assert result["criterion"] == "Phi2 angular stability"
    assert result["w_opt"] in result["candidate_w"]
    assert result["eligible"].dtype == bool
    assert np.isfinite(result["phi2"][result["best_index"]])
    assert not result["eligible"][0]


def test_discrete_api_uses_volume2_stencils_not_continuous_gradient_chain():
    delta = 0.1
    u = np.sin(np.arange(301) * delta)
    result = compute_discrete_agencity(
        u,
        delta=delta,
        A_ref=1.0,
        tau=2.0,
        w=1.0,
        P_c=3.0,
    )
    delta_star = delta / 2.0
    np.testing.assert_allclose(result.X_star, volume2_first_difference(u, delta_star))
    np.testing.assert_allclose(result.A_star, volume2_second_difference(u, delta_star))

    continuous_sampled = compute_agencity(
        u=u,
        xi=np.arange(u.size) * delta,
        A_ref=1.0,
        tau=2.0,
        w=1.0,
        P_c=3.0,
    )
    assert not np.allclose(result.A_star, continuous_sampled.A_star)
    assert result.config["formulation"] == "volume2_discrete"


def test_multivariate_aggregation_is_pc_weighted_and_flux_additive():
    xi, u1 = _signal()
    u = np.column_stack([u1, np.cos(xi)])
    result = compute_multivariate_agencity(
        u,
        xi,
        A_ref=[1.0, 2.0],
        tau=[2.0, 2.0],
        P_c=[2.0, 3.0],
    )

    np.testing.assert_allclose(result["b_total"], np.sum(result["b_components"], axis=0))
    expected_beta = (
        2.0 * result["beta_components"][0] + 3.0 * result["beta_components"][1]
    ) / 5.0
    np.testing.assert_allclose(result["beta_multi"], expected_beta)
    np.testing.assert_allclose(result["b_total"], result["P_c_total"] * result["beta_multi"])
    assert np.all(result["beta_multi_defined"])


def test_multivariate_time_varying_power_uses_pointwise_weighting():
    xi, u1 = _signal()
    u = np.column_stack([u1, np.cos(xi)])
    powers = np.column_stack([
        np.linspace(1.0, 2.0, xi.size),
        np.linspace(3.0, 4.0, xi.size),
    ])
    result = compute_multivariate_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        P_c=powers,
    )
    expected = np.sum(result["P_c_components"] * result["beta_components"], axis=0)
    np.testing.assert_allclose(result["b_total"], expected)


def test_multivariate_zero_total_power_is_explicitly_marked_undefined_for_beta_mean():
    xi, u1 = _signal()
    u = np.column_stack([u1, np.cos(xi)])
    powers = np.ones((xi.size, 2))
    powers[100:150, :] = 0.0
    result = compute_multivariate_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        P_c=powers,
    )
    np.testing.assert_array_equal(result["b_total"][100:150], 0.0j)
    np.testing.assert_array_equal(result["beta_multi"][100:150], 0.0j)
    assert not np.any(result["beta_multi_defined"][100:150])
    assert np.all(result["beta_multi_defined"][:100])


def test_riemannian_extension_is_explicitly_not_a_production_pipeline():
    status = riemannian_extension_status()
    assert status["implemented"] is False
    assert status["status"].startswith("experimental")
