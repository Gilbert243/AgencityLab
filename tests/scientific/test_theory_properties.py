import numpy as np

from agencitylab import compute_agencity
from benchmarks.scientific.reference_bench import sine_reference, structural_mask


def _compute(u, xi, tau, *, A_ref=1.0, P_c=1.0):
    return compute_agencity(u=u, xi=xi, A_ref=A_ref, tau=tau, P_c=P_c)


def test_translation_invariance_of_complete_canonical_result():
    case = sine_reference(samples_per_tau=64, cycles=6)
    baseline = case.compute()
    shifted = _compute(case.u + 7.25, case.xi, case.tau)
    mask = structural_mask(baseline)

    np.testing.assert_allclose(shifted.D[mask], baseline.D[mask], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(shifted.S[mask], baseline.S[mask], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(shifted.beta[mask], baseline.beta[mask], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(shifted.b[mask], baseline.b[mask], rtol=1e-10, atol=1e-10)


def test_global_sign_inversion_invariance():
    case = sine_reference(samples_per_tau=64, cycles=6)
    baseline = case.compute()
    inverted = _compute(-case.u, case.xi, case.tau)
    mask = structural_mask(baseline)

    np.testing.assert_allclose(inverted.M[mask], baseline.M[mask], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inverted.O[mask], baseline.O[mask], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inverted.D[mask], baseline.D[mask], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inverted.beta[mask], baseline.beta[mask], rtol=1e-11, atol=1e-11)


def test_temporal_dilation_covariance_when_tau_scales_with_time():
    case = sine_reference(samples_per_tau=64, cycles=6)
    dilation = 2.5
    baseline = case.compute()
    rescaled = _compute(
        case.u,
        case.xi / dilation,
        case.tau / dilation,
        A_ref=case.A_ref,
        P_c=case.P_c,
    )
    mask = structural_mask(baseline)

    np.testing.assert_allclose(rescaled.t_star, baseline.t_star, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(rescaled.M[mask], baseline.M[mask], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rescaled.O[mask], baseline.O[mask], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(rescaled.D[mask], baseline.D[mask], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rescaled.beta[mask], baseline.beta[mask], rtol=1e-10, atol=1e-10)


def test_arbitrarily_small_structured_signal_keeps_nonzero_state_limit():
    case = sine_reference(samples_per_tau=64, cycles=6)
    epsilons = (1e-2, 1e-4, 1e-6)
    results = [_compute(eps * case.u, case.xi, case.tau) for eps in epsilons]
    mask = structural_mask(results[0])

    for result in results[1:]:
        np.testing.assert_allclose(result.S[mask], results[0].S[mask], rtol=1e-9, atol=1e-9)

    means = np.asarray([np.mean(np.abs(result.beta[mask])) for result in results])
    assert means[-1] > 0.0
    assert abs(means[-1] - means[-2]) < abs(means[-2] - means[-3])
    assert means[-1] > 0.5 * means[0]


def test_large_amplitude_growth_is_logarithmic_with_expected_asymptotic_slope():
    case = sine_reference(samples_per_tau=64, cycles=6)
    scales = np.asarray([30.0, 100.0, 300.0, 1000.0])
    statistics = []

    for scale in scales:
        result = _compute(scale * case.u, case.xi, case.tau)
        mask = structural_mask(result)
        statistics.append(float(np.quantile(np.abs(result.beta[mask]), 0.9)))

    slope, _ = np.polyfit(np.log(scales), np.asarray(statistics), 1)
    assert 1.5 < slope < 2.5
    assert np.all(np.diff(statistics) > 0.0)


def test_characteristic_power_scales_flux_linearly_without_changing_beta():
    case = sine_reference(samples_per_tau=64, cycles=6)
    low = _compute(case.u, case.xi, case.tau, P_c=2.0)
    high = _compute(case.u, case.xi, case.tau, P_c=7.0)
    mask = structural_mask(low)

    np.testing.assert_allclose(high.beta[mask], low.beta[mask], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(high.b[mask], 3.5 * low.b[mask], rtol=1e-13, atol=1e-13)
