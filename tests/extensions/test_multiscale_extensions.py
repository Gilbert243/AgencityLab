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
    # One-sample CRM has zero empirical variance, hence no defined structural
    # orientation and must not win by an artificial zero angular variance.
    assert not result["eligible"][0]


def test_discrete_convenience_matches_explicit_coordinate_api():
    delta = 0.1
    u = np.sin(np.arange(301) * delta)
    direct = compute_discrete_agencity(
        u,
        delta=delta,
        A_ref=1.0,
        tau=2.0,
        P_c=3.0,
    )
    explicit = compute_agencity(
        u=u,
        xi=np.arange(u.size) * delta,
        A_ref=1.0,
        tau=2.0,
        P_c=3.0,
    )
    np.testing.assert_allclose(direct.b, explicit.b)
    np.testing.assert_allclose(direct.beta, explicit.beta)


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


def test_riemannian_extension_is_explicitly_not_a_production_pipeline():
    status = riemannian_extension_status()
    assert status["implemented"] is False
    assert status["status"].startswith("experimental")
