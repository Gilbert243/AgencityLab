import numpy as np

from agencitylab.analysis.regimes import (
    RegimeCriteria,
    _periodicity_diagnostic,
    classify_regime,
)


CRITERIA = RegimeCriteria(
    sigma_theta_low_max=0.2,
    sigma_theta_high_min=1.0,
    tail_cv_max=0.1,
    unstable_growth_ratio_min=2.0,
    curvature_zero_max=0.05,
    periodicity_min=0.8,
    weak_flow_max=0.2,
    weak_structure_max=0.2,
    weak_beta_variance_max=0.05,
    structure_variability_min=0.05,
)


def signature(**overrides):
    base = {
        "exact_null": False,
        "mean_abs_b": 1.0,
        "sigma_theta_mean": 0.1,
        "tail_curvature_mean_abs": 0.0,
        "growth_ratio_abs_beta": 1.0,
        "tail_beta_relative_rms": 0.5,
        "periodicity_score": 0.0,
        "variance_beta": 0.2,
        "mean_abs_beta": 1.0,
        "tail_beta_mean_abs": 1.0,
        "mean_S": 0.7,
        "std_S": 0.02,
        "tail_mean_J": 0.0,
    }
    base.update(overrides)
    return base


def test_classifier_has_no_universal_non_null_default():
    assert classify_regime(np.array([1.0 + 0.0j])) == "undetermined"
    assert classify_regime(np.zeros(5, dtype=complex)) == "undetermined"
    assert classify_regime(signature(exact_null=True)) == "null"


def test_explicit_criteria_map_theory_regime_signatures():
    assert classify_regime(
        signature(growth_ratio_abs_beta=3.0, tail_mean_J=0.5),
        criteria=CRITERIA,
    ) == "unstable"

    assert classify_regime(
        signature(tail_beta_relative_rms=0.05, tail_mean_J=-0.5),
        criteria=CRITERIA,
    ) == "passive_damped"

    assert classify_regime(
        signature(
            periodicity_score=0.95,
            tail_beta_relative_rms=0.4,
            tail_curvature_mean_abs=0.0,
        ),
        criteria=CRITERIA,
    ) == "active_oscillating"

    assert classify_regime(
        signature(
            sigma_theta_mean=2.0,
            mean_S=0.1,
            std_S=0.02,
            variance_beta=0.02,
        ),
        criteria=CRITERIA,
    ) == "stochastic"

    assert classify_regime(
        signature(
            sigma_theta_mean=2.0,
            mean_S=0.6,
            std_S=0.2,
            variance_beta=0.4,
            periodicity_score=0.3,
        ),
        criteria=CRITERIA,
    ) == "chaotic"


def test_absolute_flow_magnitude_no_longer_splits_noise_from_chaos():
    noisy = signature(
        sigma_theta_mean=2.0,
        mean_abs_b=1000.0,
        mean_S=0.1,
        variance_beta=0.02,
    )
    chaotic = signature(
        sigma_theta_mean=2.0,
        mean_abs_b=0.001,
        mean_S=0.6,
        std_S=0.2,
        variance_beta=0.4,
        periodicity_score=0.3,
    )
    assert classify_regime(noisy, criteria=CRITERIA) == "stochastic"
    assert classify_regime(chaotic, criteria=CRITERIA) == "chaotic"


def test_high_sigma_is_undetermined_without_structure_variance_criteria():
    legacy_criteria = RegimeCriteria(
        sigma_theta_low_max=0.2,
        sigma_theta_high_min=1.0,
        tail_cv_max=0.1,
        unstable_growth_ratio_min=2.0,
        curvature_zero_max=0.05,
        periodicity_min=0.8,
        weak_flow_max=0.2,
    )
    assert classify_regime(
        signature(sigma_theta_mean=2.0, mean_abs_b=0.01),
        criteria=legacy_criteria,
    ) == "undetermined"


def test_periodicity_is_estimated_independently_of_tau():
    xi = np.linspace(0.0, 40.0, 4001)
    period = 2.5
    beta = (0.7 + 0.3 * np.cos(2.0 * np.pi * xi / period)) + 0.1j * np.sin(
        2.0 * np.pi * xi / period
    )

    score, estimated_period = _periodicity_diagnostic(beta, xi)

    assert score > 0.99
    assert estimated_period is not None
    assert np.isclose(estimated_period, period, rtol=0.01)


def test_periodicity_is_undefined_for_a_fixed_point():
    xi = np.linspace(0.0, 20.0, 1001)
    beta = np.full(xi.shape, -0.5 + 0.0j)
    score, estimated_period = _periodicity_diagnostic(beta, xi)
    assert np.isnan(score)
    assert estimated_period is None


def test_exact_null_requires_absence_of_dynamics_not_only_zero_flux(monkeypatch):
    import agencitylab.analysis.regimes as regimes

    class Interval:
        mask = np.ones(101, dtype=bool)
        valid_fraction = 1.0
        start_time = 0.0
        stop_time = 10.0
        memory_window = 1.0
        memory_window_source = "tau_fallback"

    monkeypatch.setattr(regimes, "resolve_analysis_interval", lambda *args, **kwargs: Interval())
    monkeypatch.setattr(
        regimes,
        "sigma_theta",
        lambda theta, xi, tau, valid_mask=None: np.full(theta.shape, np.nan),
    )
    monkeypatch.setattr(
        regimes,
        "curvature",
        lambda beta, xi: np.full(beta.shape, np.nan),
    )
    monkeypatch.setattr(
        regimes,
        "detect_agencity_zeros",
        lambda S, J: np.flatnonzero((S == 0.0) | (J == 0.0)),
    )

    xi = np.linspace(0.0, 10.0, 101)
    zeros = np.zeros_like(xi)
    rest = {
        "xi": xi,
        "tau": 1.0,
        "D": zeros,
        "S": zeros,
        "J": zeros,
        "theta": zeros,
        "beta": zeros.astype(complex),
        "b": zeros.astype(complex),
    }
    dynamic_without_structure = {
        **rest,
        "D": np.ones_like(xi),
        "J": np.full_like(xi, np.log((np.e + 1.0) / np.e)),
    }

    assert regimes.regime_signature(rest)["exact_null"] is True
    assert regimes.regime_signature(dynamic_without_structure)["exact_null"] is False
