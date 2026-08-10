import json

import numpy as np

from agencitylab import (
    RegimeCriteria,
    analyze_agencity,
    analyze_coherence,
    analyze_events,
    analyze_geometry,
    analyze_regime,
    analyze_transitions,
    compute_agencity,
)


def _sine_result(P_c=1.0):
    xi = np.linspace(0.0, 20.0, 201)
    return compute_agencity(
        u=np.sin(xi),
        xi=xi,
        A_ref=1.0,
        tau=2.0,
        P_c=P_c,
    )


def test_analysis_does_not_mutate_canonical_arrays():
    result = _sine_result()
    fields = ("X_star", "A_star", "M", "O", "D", "S", "J", "U", "beta", "b", "theta")
    before = {name: np.array(getattr(result, name), copy=True) for name in fields}

    report = analyze_agencity(result)

    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(result, name), expected)
    assert report["analysis_schema_version"] == "0.5"
    assert report["regime"] == "undetermined"
    assert report["real_agencity"]["status"] == "undetermined"
    assert report["geometry"]["geometry_source"] == "beta"


def test_real_agencity_analysis_requires_explicit_interpretive_thresholds():
    result = _sine_result()

    default = analyze_coherence(result)
    configured = analyze_coherence(
        result,
        real_agencity_thresholds={
            "theta_variance_threshold": 10.0,
            "b_threshold": 0.0,
            "min_fraction": 0.1,
        },
    )

    assert default["real_agencity"]["real_agencity"] is None
    assert configured["real_agencity"]["status"] == "local and global criterion evaluated"
    assert configured["real_agencity"]["thresholds"]["Sigma_Theta_max"] == 10.0


def test_geometry_is_intrinsic_to_beta_not_time_varying_characteristic_power():
    base = _sine_result(P_c=1.0)
    profile = np.linspace(1.0, 3.0, base.xi.size)
    scaled = _sine_result(P_c=profile)

    np.testing.assert_allclose(base.beta, scaled.beta)
    assert not np.allclose(base.b, scaled.b)

    geometry_base = analyze_geometry(base)
    geometry_scaled = analyze_geometry(scaled)
    np.testing.assert_allclose(
        geometry_base["curvature"],
        geometry_scaled["curvature"],
        equal_nan=True,
    )


def test_regime_api_defaults_to_undetermined_for_non_null_result():
    result = _sine_result()

    assert analyze_regime(result) == "undetermined"


def test_regime_criteria_are_serialized_in_structured_report():
    result = _sine_result()
    criteria = RegimeCriteria(
        sigma_theta_low_max=0.2,
        sigma_theta_high_min=1.0,
        tail_cv_max=0.2,
        unstable_growth_ratio_min=2.0,
        curvature_zero_max=0.05,
        periodicity_min=0.8,
        weak_flow_max=0.2,
    )

    report = analyze_agencity(result, regime_criteria=criteria)

    assert isinstance(report["regime_classification"]["criteria"], dict)
    json.dumps(report)


def test_event_and_transition_api_preserve_pre_v05_summary_keys():
    result = _sine_result()

    events = analyze_events(result)
    transitions = analyze_transitions(result)

    assert "event_count" in events
    assert "event_indices" in events
    assert "dynamic_peaks" in events
    assert "transition_count" in transitions
    assert "transition_indices" in transitions
    assert "zeros" in transitions
    assert "critical_surface_D_equals_S" in transitions
