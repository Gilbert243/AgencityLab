import numpy as np
import pytest

import agencitylab.biology as bio


def _reference():
    return bio.BiologicalReference(
        taxon="species-x",
        support="whole organism",
        causal_input="challenge",
        causal_output="response",
        identifier="ref-x",
    )


def test_specific_metabolic_reference_uses_median_power_per_mass():
    powers = np.array([100.0, 240.0, 450.0])
    masses = np.array([10.0, 20.0, 30.0])
    assert bio.specific_metabolic_reference(powers, masses) == pytest.approx(12.0)


def test_biological_reference_time_follows_closed_mapping():
    result = bio.biological_reference_time(
        [100.0, 240.0, 450.0],
        [10.0, 20.0, 30.0],
        reference=_reference(),
    )
    assert result.q_reference == pytest.approx(12.0)
    assert result.tau == pytest.approx(1.0 / 12.0)
    assert result.sample_size == 3


def test_reference_time_rejects_nonphysical_input():
    with pytest.raises(ValueError, match="masses must be strictly positive"):
        bio.biological_reference_time([10.0, 20.0], [1.0, 0.0], reference=_reference())


def test_causal_time_recovers_monoexponential_limit():
    nu = 0.4
    t = np.linspace(0.0, 40.0, 40001)
    h = 3.7 * np.exp(-nu * t)
    assert bio.causal_time(t, h) == pytest.approx(1.0 / nu, rel=2e-5)


def test_causal_time_is_invariant_to_impulse_amplitude():
    t = np.linspace(0.0, 30.0, 30001)
    base = np.exp(-0.5 * t)
    assert bio.causal_time(t, base) == pytest.approx(bio.causal_time(t, 17.0 * base))


def test_causal_time_recovers_delay_plus_relaxation():
    delay = 3.0
    tau_r = 2.5
    t = np.linspace(0.0, 50.0, 50001)
    h = np.where(t >= delay, np.exp(-(t - delay) / tau_r), 0.0)
    assert bio.causal_time(t, h) == pytest.approx(delay + tau_r, rel=1e-4)


def test_causal_time_rejects_zero_response():
    t = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="non-zero absolute area"):
        bio.causal_time(t, np.zeros_like(t))


def test_memory_reference_uses_population_mean_not_median():
    t = np.linspace(0.0, 80.0, 80001)
    responses = [
        (t, np.exp(-t / 1.0)),
        (t, np.exp(-t / 2.0)),
        (t, np.exp(-t / 9.0)),
    ]
    result = bio.calibrate_memory_reference(responses, reference=_reference())
    assert result.w == pytest.approx((1.0 + 2.0 + 9.0) / 3.0, rel=2e-3)
    assert result.w != pytest.approx(2.0, rel=1e-2)
