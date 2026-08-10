import numpy as np

from agencitylab.analysis.coherence import (
    detect_structural_plateaus,
    real_agencity_criterion,
    sigma_theta,
)


def test_sigma_theta_uses_complete_tau_window_and_local_unwrap():
    xi = np.arange(5.0)
    theta = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    sigma = sigma_theta(theta, xi, tau=2.0)

    assert np.isnan(sigma[0])
    assert np.isnan(sigma[1])
    np.testing.assert_allclose(sigma[2], np.var([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(sigma[4], np.var([0.2, 0.3, 0.4]))


def test_sigma_theta_does_not_bridge_undefined_structural_orientation():
    xi = np.arange(6.0)
    theta = np.linspace(0.0, 0.5, 6)
    valid = np.ones(6, dtype=bool)
    valid[1] = False

    sigma = sigma_theta(theta, xi, tau=2.0, valid_mask=valid)

    assert np.isnan(sigma[2])
    assert np.isnan(sigma[3])
    assert np.isfinite(sigma[4])


def test_sigma_theta_handles_branch_cut_by_local_unwrap():
    xi = np.arange(3.0)
    theta = np.array([3.10, -3.12, -3.05])

    sigma = sigma_theta(theta, xi, tau=2.0)

    unwrapped = np.unwrap(theta)
    np.testing.assert_allclose(sigma[2], np.var(unwrapped))
    assert sigma[2] < 0.02


def test_real_agencity_requires_contextual_thresholds():
    S = np.ones(5)
    sigma = np.zeros(5)
    b = np.ones(5, dtype=complex)

    diagnostic = real_agencity_criterion(S, sigma, b)

    assert diagnostic["status"] == "undetermined"
    assert diagnostic["real_agencity"] is None
    assert diagnostic["local_real_agencity"] is None


def test_real_agencity_returns_local_mask_and_optional_global_rule():
    S = np.ones(5)
    sigma = np.array([0.1, 0.2, 0.9, 0.1, 0.2])
    b = np.array([2.0, 2.0, 2.0, 0.1, 2.0], dtype=complex)

    diagnostic = real_agencity_criterion(
        S,
        sigma,
        b,
        theta_variance_threshold=0.5,
        b_threshold=1.0,
        min_fraction=0.5,
    )

    np.testing.assert_array_equal(
        diagnostic["local_real_agencity"],
        np.array([True, True, False, False, True]),
    )
    np.testing.assert_allclose(diagnostic["real_agencity_fraction"], 3.0 / 5.0)
    assert diagnostic["real_agencity"] is True


def test_structural_plateaus_are_explicit_diagnostics():
    xi = np.arange(7.0)
    S = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    plateaus = detect_structural_plateaus(
        S,
        xi,
        slope_threshold=0.0,
        min_duration=1.0,
    )

    assert plateaus
    assert all(item["duration"] >= 1.0 for item in plateaus)
