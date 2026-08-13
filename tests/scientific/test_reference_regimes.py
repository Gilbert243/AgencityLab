import numpy as np
import pytest

from benchmarks.scientific.reference_bench import (
    circular_variance,
    periodic_relative_error,
    structural_mask,
)


def test_rest_reference_is_exactly_null(scientific_results):
    result = scientific_results["rest"]
    for name in ("X_star", "A_star", "M", "O", "D", "S", "J"):
        np.testing.assert_array_equal(getattr(result, name), 0.0)
    np.testing.assert_array_equal(result.U, 0.0j)
    np.testing.assert_array_equal(result.beta, 0.0j)
    np.testing.assert_array_equal(result.b, 0.0j)


def test_sine_reference_is_periodic_after_crm_warmup(scientific_cases, scientific_results):
    case = scientific_cases["sine"]
    result = scientific_results["sine"]

    assert periodic_relative_error(result, case.expected_period) < 2e-3
    mask = structural_mask(result)
    assert np.mean(result.S[mask]) > 0.9
    assert circular_variance(result) < 1e-3


def test_damped_reference_becomes_structure_dominated_with_residual_state(
    scientific_cases, scientific_results
):
    case = scientific_cases["damped"]
    result = scientific_results["damped"]
    samples_per_period = int(round(case.tau / np.median(np.diff(case.xi))))
    first = slice(2 * samples_per_period, 3 * samples_per_period)
    last = slice(-samples_per_period - 1, -1)

    assert np.mean(result.D[last]) < 0.1 * np.mean(result.D[first])
    assert np.mean(result.J[last]) < 0.0
    assert np.mean(np.abs(result.beta[last])) > 1e-3
    assert circular_variance(result) < 0.05


def test_van_der_pol_reference_is_bounded_and_late_periodic(scientific_cases, scientific_results):
    case = scientific_cases["van_der_pol"]
    result = scientific_results["van_der_pol"]
    mask = structural_mask(result)

    assert np.all(np.isfinite(result.b[mask]))
    assert np.std(np.abs(result.beta[mask])) > 0.0
    assert periodic_relative_error(result, case.expected_period, tail_fraction=0.35) < 0.2
    assert np.max(np.abs(result.beta[mask])) < 20.0


def test_unstable_reference_has_asymptotically_linear_logarithmic_contrast(
    scientific_cases, scientific_results
):
    case = scientific_cases["unstable"]
    result = scientific_results["unstable"]
    samples_per_period = int(round(case.tau / np.median(np.diff(case.xi))))

    block_means = []
    block_times = []
    for block in range(4, 10):
        start = block * samples_per_period
        stop = (block + 1) * samples_per_period
        if stop > len(result.J):
            break
        block_means.append(float(np.mean(result.J[start:stop])))
        block_times.append(float(np.mean(result.xi[start:stop])))

    block_means = np.asarray(block_means)
    block_times = np.asarray(block_times)
    slope, intercept = np.polyfit(block_times, block_means, 1)
    fitted = slope * block_times + intercept
    ss_res = float(np.sum((block_means - fitted) ** 2))
    ss_tot = float(np.sum((block_means - np.mean(block_means)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot

    assert slope > 0.0
    assert slope == pytest.approx(0.2, rel=0.35)
    assert r_squared > 0.95
    assert block_means[-1] > block_means[0]


def test_filtered_ou_is_dynamic_with_irregular_orientation(scientific_results):
    sine = scientific_results["sine"]
    ou = scientific_results["filtered_ou"]
    mask = structural_mask(ou)

    assert np.all(np.isfinite(ou.b[mask]))
    assert np.any(ou.D[mask] > 0.0)
    assert circular_variance(ou) > circular_variance(sine) + 0.1


def test_lorenz_reference_is_bounded_on_fixed_window_with_irregular_orientation(
    scientific_results,
):
    sine = scientific_results["sine"]
    lorenz = scientific_results["lorenz"]
    mask = structural_mask(lorenz)

    assert np.all(np.isfinite(lorenz.b[mask]))
    assert np.max(np.abs(lorenz.beta[mask])) < 100.0
    assert np.std(np.abs(lorenz.beta[mask])) > 0.0
    assert circular_variance(lorenz) > circular_variance(sine) + 0.1
