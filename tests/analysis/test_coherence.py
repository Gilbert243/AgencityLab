import numpy as np

from agencitylab.analysis.coherence import (
    detect_structural_plateaus,
    real_agencity_criterion,
    sigma_theta,
)


def _direct_sigma_theta(theta, xi, tau, valid_mask=None):
    """Evaluate the accepted local-unwrapped variance definition directly."""
    theta = np.asarray(theta, dtype=float)
    xi = np.asarray(xi, dtype=float)
    valid = np.ones(theta.size, dtype=bool) if valid_mask is None else valid_mask
    out = np.full(theta.size, np.nan, dtype=float)

    for index, time_value in enumerate(xi):
        if time_value - xi[0] < tau:
            continue
        left = int(np.searchsorted(xi, time_value - tau, side="left"))
        if index - left + 1 < 2 or not np.all(valid[left : index + 1]):
            continue
        out[index] = float(np.var(np.unwrap(theta[left : index + 1])))
    return out


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


def test_optimized_sigma_theta_matches_direct_definition_on_irregular_axis():
    rng = np.random.default_rng(7129)
    xi = np.cumsum(rng.uniform(0.04, 0.12, size=2048))
    unwrapped = 0.31 * xi + 0.8 * np.sin(0.17 * xi) + 0.03 * np.sin(1.7 * xi)
    theta = np.angle(np.exp(1j * unwrapped))
    valid = np.ones(theta.size, dtype=bool)
    valid[311:315] = False
    valid[1007] = False
    valid[1700:1703] = False
    tau = 2.25

    actual = sigma_theta(theta, xi, tau, valid_mask=valid)
    expected = _direct_sigma_theta(theta, xi, tau, valid_mask=valid)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    # Prefix-moment subtraction accumulates a bounded floating-point difference
    # while preserving the accepted interval, unwrap, and population variance.
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-9, atol=2e-11)


def test_sigma_theta_constant_orientation_is_exact_zero_after_complete_window():
    xi = np.linspace(0.0, 10.0, 201)
    theta = np.full(xi.size, 1.23456789)

    sigma = sigma_theta(theta, xi, tau=1.0)
    finite = np.isfinite(sigma)

    assert np.any(finite)
    np.testing.assert_array_equal(sigma[finite], 0.0)


def test_sigma_theta_long_high_winding_signal_is_finite_and_nonnegative():
    size = 100_000
    xi = np.arange(size, dtype=float) * 0.05
    unwrapped = 0.8 * xi + 0.1 * np.sin(0.11 * xi)
    theta = np.angle(np.exp(1j * unwrapped))

    sigma = sigma_theta(theta, xi, tau=2.0)
    finite = np.isfinite(sigma)

    assert sigma.shape == theta.shape
    assert np.count_nonzero(finite) > size - 100
    assert np.all(sigma[finite] >= 0.0)


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
