import numpy as np

from agencitylab.analysis.regimes import RegimeCriteria, classify_regime


CRITERIA = RegimeCriteria(
    sigma_theta_low_max=0.2,
    sigma_theta_high_min=1.0,
    tail_cv_max=0.1,
    unstable_growth_ratio_min=2.0,
    curvature_zero_max=0.05,
    periodicity_min=0.8,
    weak_flow_max=0.2,
)


def signature(**overrides):
    base = {
        "exact_null": False,
        "sigma_theta_mean": 0.1,
        "curvature_mean_abs": 0.0,
        "growth_ratio_abs_b": 1.0,
        "tail_cv_abs_b": 0.5,
        "tau_periodicity_score": 0.0,
        "mean_abs_b": 1.0,
        "tail_mean_J": 0.0,
    }
    base.update(overrides)
    return base


def test_classifier_has_no_universal_non_null_default():
    assert classify_regime(np.array([1.0 + 0.0j])) == "undetermined"
    assert classify_regime(np.zeros(5, dtype=complex)) == "null"


def test_explicit_criteria_map_theory_regime_signatures():
    assert classify_regime(
        signature(growth_ratio_abs_b=3.0),
        criteria=CRITERIA,
    ) == "unstable"

    assert classify_regime(
        signature(tail_cv_abs_b=0.05, tail_mean_J=-0.5),
        criteria=CRITERIA,
    ) == "passive_damped"

    assert classify_regime(
        signature(curvature_mean_abs=0.2, tau_periodicity_score=0.95),
        criteria=CRITERIA,
    ) == "active_oscillating"

    assert classify_regime(
        signature(sigma_theta_mean=2.0, mean_abs_b=0.1),
        criteria=CRITERIA,
    ) == "stochastic"

    assert classify_regime(
        signature(sigma_theta_mean=2.0, mean_abs_b=1.0),
        criteria=CRITERIA,
    ) == "chaotic"
