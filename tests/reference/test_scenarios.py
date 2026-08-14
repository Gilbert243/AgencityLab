from __future__ import annotations

import numpy as np
import pytest

from agencitylab.models import AgencityResult
from agencitylab.reference import scenarios


FACTORIES = (
    scenarios.rest,
    scenarios.sinusoidal,
    scenarios.damped,
    scenarios.van_der_pol,
    scenarios.unstable,
    scenarios.stochastic,
    scenarios.lorenz,
)


def test_scenario_catalog_has_all_historical_reference_sources():
    assert scenarios.available() == (
        "rest",
        "sinusoidal",
        "damped",
        "van_der_pol",
        "unstable",
        "stochastic",
        "lorenz",
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_scenarios_expose_observable_and_explicit_physical_context(factory):
    scenario = factory()

    assert scenario.u is scenario.signal.u
    assert scenario.xi is scenario.signal.xi
    assert scenario.A_ref > 0.0
    assert scenario.tau > 0.0
    assert scenario.w > 0.0
    assert scenario.P_c >= 0.0
    assert scenario.regime
    assert scenario.description
    assert scenario.source_theory
    assert scenario.scientific_status == "reproducible scientific reference setup"
    assert scenario.context["regime"] == scenario.regime


def test_compute_delegates_to_public_canonical_pipeline(monkeypatch):
    scenario = scenarios.sinusoidal(samples_per_tau=16, cycles=3, P_c=2.5)
    sentinel = object()
    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(scenarios, "compute_agencity", fake_compute)
    assert scenario.compute() is sentinel
    np.testing.assert_array_equal(captured["u"], scenario.u)
    np.testing.assert_array_equal(captured["xi"], scenario.xi)
    assert captured["A_ref"] == scenario.A_ref
    assert captured["tau"] == scenario.tau
    assert captured["w"] == scenario.w
    assert captured["P_c"] == scenario.P_c
    assert captured["metadata"].extra["source_regime"] == scenario.regime


@pytest.mark.parametrize("factory", FACTORIES)
def test_compute_returns_finite_canonical_result_with_matching_dimensions(factory):
    scenario = factory()
    result = scenario.compute()

    assert isinstance(result, AgencityResult)
    assert result.beta.shape == scenario.u.shape
    assert result.b.shape == scenario.u.shape
    assert np.all(np.isfinite(result.beta))
    assert np.all(np.isfinite(result.b))


def test_invalid_scenario_context_is_rejected():
    signal = scenarios.rest().signal
    with pytest.raises(ValueError, match="A_ref"):
        scenarios.ReferenceScenario(
            name="invalid",
            signal=signal,
            A_ref=0.0,
            tau=1.0,
            w=1.0,
            P_c=1.0,
            regime="invalid",
            description="invalid",
        )
