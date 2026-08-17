import math
from types import SimpleNamespace

import numpy as np
import pytest

from agencitylab import compute_agencity
from agencitylab.analysis.geometry import (
    curvature,
    geometric_summary,
    net_phase_turns,
    winding_number,
)
from agencitylab.analysis.information.landauer import (
    LN2,
    landauer_from_entropy,
    landauer_lower_bound,
)
from agencitylab.analysis.regimes import regime_signature
from agencitylab.analysis.reports import build_report_dict
from agencitylab.analysis.validity import resolve_analysis_interval
from agencitylab.constants.physics import BOLTZMANN_CONSTANT


def _result(*, magnitude, w=2.0):
    xi = np.linspace(0.0, 20.0, 201)
    magnitude = np.asarray(magnitude, dtype=float)
    if magnitude.ndim == 0 or magnitude.size == 1:
        magnitude = np.full(xi.size, magnitude.reshape(-1)[0], dtype=float)
    elif magnitude.ndim != 1 or magnitude.size != xi.size:
        raise ValueError("magnitude must be scalar, size-1, or match xi")
    return SimpleNamespace(
        xi=xi,
        tau=1.0,
        memory_window=w,
        S=np.ones(xi.size),
        D=np.ones(xi.size),
        J=np.zeros(xi.size),
        theta=np.zeros(xi.size),
        beta=magnitude.astype(complex),
        b=magnitude.astype(complex),
    )


def test_landauer_uses_numeric_constant_value():
    expected = BOLTZMANN_CONSTANT.value * 300.0 * LN2
    assert landauer_lower_bound(1.0, 300.0) == pytest.approx(expected)
    assert landauer_lower_bound(1000.0, 300.0) == pytest.approx(1000.0 * expected)
    assert landauer_lower_bound(0.0, 300.0) == 0.0
    assert landauer_lower_bound(1.0, 0.0) == 0.0
    assert landauer_from_entropy(LN2, 300.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("bits", "temperature"),
    [
        (-1.0, 300.0),
        (1.0, -1.0),
        (math.nan, 1.0),
        (1.0, math.inf),
        (True, 300.0),
    ],
)
def test_landauer_rejects_invalid_inputs(bits, temperature):
    with pytest.raises(ValueError):
        landauer_lower_bound(bits, temperature)


def test_landauer_from_entropy_rejects_invalid_entropy():
    with pytest.raises(ValueError):
        landauer_from_entropy(-1.0, 300.0)


def test_curvature_marks_numerically_stationary_trajectory_undefined():
    xi = np.linspace(0.0, 10.0, 401)
    beta = 1.0 + 1.0j + 1e-15 * xi + 1e-15j * xi**2
    assert np.all(np.isnan(curvature(beta, xi)))
    summary = geometric_summary(beta, xi=xi)
    assert summary["curvature_defined_fraction"] == 0.0
    assert np.isnan(summary["curvature_mean_abs"])


def test_curvature_line_and_circle_remain_correct():
    xi = np.linspace(0.0, 2.0 * np.pi, 1001)
    line = (2.0 * xi + 1.0j * xi).astype(complex)
    line_curvature = curvature(line, xi)
    np.testing.assert_allclose(line_curvature[np.isfinite(line_curvature)], 0.0, atol=1e-10)

    circle = 2.0 * np.exp(1j * xi)
    circle_curvature = curvature(circle, xi)
    np.testing.assert_allclose(np.median(circle_curvature[5:-5]), 0.5, rtol=2e-3, atol=2e-3)


def test_open_phase_turns_are_not_silently_called_winding():
    theta = np.linspace(0.0, 1.5 * np.pi, 101)
    assert net_phase_turns(theta) == pytest.approx(0.75)
    assert np.isnan(winding_number(theta, closed=False))


def test_analysis_interval_uses_memory_window_not_tau():
    result = _result(magnitude=np.array([1.0]), w=2.0)
    interval = resolve_analysis_interval(result, edge_samples=1)
    assert interval.memory_window == 2.0
    assert interval.memory_window_source == "memory_window"
    assert interval.start_time == pytest.approx(4.0)
    assert not interval.mask[-1]


def test_regime_growth_excludes_crm_warmup():
    xi = np.linspace(0.0, 20.0, 201)
    stationary = np.where(xi < 4.0, 0.0, 1.0)
    signature = regime_signature(_result(magnitude=stationary, w=2.0))
    assert signature["growth_ratio_abs_b"] == pytest.approx(1.0)
    assert signature["analysis_window"] == 2.0


def test_regime_growth_distinguishes_decay_and_growth_after_warmup():
    xi = np.linspace(0.0, 20.0, 201)
    elapsed = np.maximum(0.0, xi - 4.0)
    decay = np.where(xi < 4.0, 0.0, np.exp(-0.1 * elapsed))
    growth = np.where(xi < 4.0, 0.0, np.exp(0.1 * elapsed))
    assert regime_signature(_result(magnitude=decay))["growth_ratio_abs_b"] < 1.0
    assert regime_signature(_result(magnitude=growth))["growth_ratio_abs_b"] > 1.0


def test_exact_null_periodicity_is_undefined():
    result = _result(magnitude=np.array([0.0]))
    result.S = np.zeros_like(result.S)
    signature = regime_signature(result)
    assert signature["exact_null"] is True
    assert np.isnan(signature["tau_periodicity_score"])
    assert signature["tau_periodicity_defined"] is False


def test_result_summary_adds_explicit_magnitude_and_circular_names():
    xi = np.arange(0.0, 8.0 + 0.1, 0.1)
    result = compute_agencity(
        np.sin(2.0 * np.pi * xi),
        xi,
        A_ref=1.0,
        tau=1.0,
        w=1.0,
        P_c=2.0,
    )
    summary = result.summary()
    assert summary["mean_abs_b"] == summary["b_mean"]
    assert summary["mean_abs_beta"] == summary["beta_mean"]
    assert np.isfinite(summary["theta_circular_mean"])
    assert -1e-12 <= summary["theta_circular_variance"] <= 1.0 + 1e-12

    rest = compute_agencity(
        np.ones_like(xi),
        xi,
        A_ref=1.0,
        tau=1.0,
        w=1.0,
        P_c=1.0,
    )
    assert np.isnan(rest.summary()["theta_circular_mean"])


def test_report_uses_shared_memory_window_interval():
    xi = np.arange(0.0, 12.0 + 0.1, 0.1)
    result = compute_agencity(
        np.sin(2.0 * np.pi * xi),
        xi,
        A_ref=1.0,
        tau=1.0,
        w=2.0,
        P_c=1.0,
    )
    report = build_report_dict(result)
    interval = report["analysis_interval"]
    assert interval["memory_window"] == 2.0
    assert interval["finite_record_crm_start_time"] == pytest.approx(4.0)
    assert report["geometry"]["winding"]["defined"] is False
    assert "net_phase_turns" in report["geometry"]
