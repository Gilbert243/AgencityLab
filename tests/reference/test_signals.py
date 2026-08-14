from __future__ import annotations

import numpy as np
import pytest

from agencitylab.models import AgencitySignal
from agencitylab.reference import signals


DETERMINISTIC = (
    signals.constant,
    signals.sinusoid,
    signals.damped_oscillator,
    signals.van_der_pol,
    signals.unstable_oscillator,
    signals.lorenz,
)


@pytest.mark.parametrize("generator", DETERMINISTIC)
def test_deterministic_generators_are_repeatable_finite_signals(generator):
    first = generator()
    second = generator()

    assert isinstance(first, AgencitySignal)
    assert first.xi.ndim == first.u.ndim == 1
    assert first.xi.shape == first.u.shape
    assert first.xi.size > 2
    assert np.all(np.diff(first.xi) > 0.0)
    assert np.all(np.isfinite(first.xi))
    assert np.all(np.isfinite(first.u))
    np.testing.assert_array_equal(first.xi, second.xi)
    np.testing.assert_array_equal(first.u, second.u)


@pytest.mark.parametrize(
    "generator",
    (signals.white_noise, signals.ornstein_uhlenbeck, signals.smoothed_ornstein_uhlenbeck),
)
def test_stochastic_generators_use_reproducible_local_rng(generator):
    first = generator(seed=123)
    repeated = generator(seed=123)
    different = generator(seed=124)

    np.testing.assert_array_equal(first.u, repeated.u)
    assert not np.array_equal(first.u, different.u)


def test_raw_and_regularized_stochastic_observables_are_explicitly_distinct():
    raw_white = signals.white_noise(seed=7)
    raw_ou = signals.ornstein_uhlenbeck(seed=7)
    regularized = signals.smoothed_ornstein_uhlenbeck(seed=7)

    assert raw_white.metadata.extra["canonical_ready"] is False
    assert raw_ou.metadata.extra["canonical_ready"] is False
    assert regularized.metadata.extra["canonical_ready"] is True
    assert "Gaussian" in regularized.metadata.extra["regularization"]
    assert not np.array_equal(raw_ou.u, regularized.u)


def test_generators_do_not_call_canonical_compute(monkeypatch):
    import agencitylab.api.compute as compute_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("signal generators must not compute Agencity")

    monkeypatch.setattr(compute_module, "compute_agencity", forbidden)
    for generator in DETERMINISTIC:
        assert generator().u.size > 0


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: signals.constant(samples_per_tau=2), "samples_per_tau"),
        (lambda: signals.sinusoid(amplitude=0.0), "amplitude"),
        (lambda: signals.damped_oscillator(damping_ratio=1.0), "damping_ratio"),
        (lambda: signals.unstable_oscillator(growth_rate=1.0), "growth_rate"),
        (lambda: signals.white_noise(scale=-1.0), "scale"),
        (lambda: signals.ornstein_uhlenbeck(theta=0.0), "theta"),
        (lambda: signals.smoothed_ornstein_uhlenbeck(smoothing_sigma_samples=0.0), "smoothing"),
        (lambda: signals.lorenz(observable="q"), "observable"),
    ),
)
def test_generator_parameter_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_available_lists_only_public_generators():
    assert set(signals.available()) == {
        "constant",
        "sinusoid",
        "damped_oscillator",
        "van_der_pol",
        "unstable_oscillator",
        "white_noise",
        "ornstein_uhlenbeck",
        "smoothed_ornstein_uhlenbeck",
        "lorenz",
    }


def test_lorenz_rk4_trajectory_converges_under_output_refinement():
    coarse = signals.lorenz(
        burn_cycles=0,
        output_cycles=4,
        samples_per_tau=25,
    )
    fine = signals.lorenz(
        burn_cycles=0,
        output_cycles=4,
        samples_per_tau=50,
    )

    np.testing.assert_allclose(coarse.xi, fine.xi[::2], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(coarse.u, fine.u[::2], rtol=0.0, atol=2e-3)
