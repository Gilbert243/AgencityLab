"""Reproducible scientific scenarios built around canonical computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agencitylab.api.compute import compute_agencity
from agencitylab.models import AgencityResult, AgencitySignal

from . import signals

_SOURCE = "Theory of Agencity, Chapters 16-19; AgencityLab scientific reference battery"
_SCENARIOS = (
    "rest",
    "sinusoidal",
    "damped",
    "van_der_pol",
    "unstable",
    "stochastic",
    "lorenz",
)


def _required_characteristic_time(signal: AgencitySignal) -> float:
    value = signal.metadata.characteristic_time
    if value is None:
        raise ValueError("reference signal metadata must define characteristic_time")
    return float(value)


@dataclass(frozen=True, slots=True)
class ReferenceScenario:
    """Observable plus explicit physical context for canonical analysis.

    ``regime`` describes the source dynamics.  It is not a forced diagnosis of
    real Agencity and it is never passed to the canonical equations.
    """

    name: str
    signal: AgencitySignal
    A_ref: float
    tau: float
    w: float
    P_c: float
    regime: str
    description: str
    source_theory: str = _SOURCE
    scientific_status: str = "reproducible scientific reference setup"

    def __post_init__(self) -> None:
        for field_name in ("A_ref", "tau", "w"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and strictly positive")
        power = float(self.P_c)
        if not np.isfinite(power) or power < 0.0:
            raise ValueError("P_c must be finite and non-negative")

    @property
    def xi(self) -> np.ndarray:
        """Coordinate samples of the scenario observable."""

        return self.signal.xi

    @property
    def u(self) -> np.ndarray:
        """Observable samples analyzed by the scenario."""

        return self.signal.u

    @property
    def expected_period(self) -> float | None:
        """Documented source period, when the generator defines one."""

        value = self.signal.metadata.extra.get("expected_period")
        return None if value is None else float(value)

    @property
    def context(self) -> dict[str, Any]:
        """Return the explicit physical and descriptive context."""

        return {
            "A_ref": self.A_ref,
            "tau": self.tau,
            "w": self.w,
            "P_c": self.P_c,
            "regime": self.regime,
            "source_theory": self.source_theory,
            "scientific_status": self.scientific_status,
        }

    def compute(self) -> AgencityResult:
        """Delegate analysis to the public canonical ``compute_agencity`` API."""

        metadata = self.signal.metadata.with_updates(
            reference_amplitude=self.A_ref,
            characteristic_time=self.tau,
            memory_window=self.w,
            characteristic_power=self.P_c,
            extra={
                **self.signal.metadata.extra,
                "reference_scenario": self.name,
                "source_regime": self.regime,
                "scenario_scientific_status": self.scientific_status,
                "source_theory": self.source_theory,
            },
        )
        return compute_agencity(
            u=self.signal.u,
            xi=self.signal.xi,
            A_ref=self.A_ref,
            tau=self.tau,
            w=self.w,
            P_c=self.P_c,
            metadata=metadata,
        )


def available() -> tuple[str, ...]:
    """Return the public reference scenario names."""

    return _SCENARIOS


def rest(*, samples_per_tau: int = 64, P_c: float = 1.0) -> ReferenceScenario:
    """Return the exact sampled rest-state scenario."""

    signal = signals.constant(samples_per_tau=samples_per_tau)
    return ReferenceScenario(
        name="rest",
        signal=signal,
        A_ref=1.0,
        tau=1.0,
        w=1.0,
        P_c=P_c,
        regime="rest / constant observable",
        description="Exact constant sampled rest state.",
    )


def sinusoidal(
    *, samples_per_tau: int = 64, cycles: int = 10, P_c: float = 1.0
) -> ReferenceScenario:
    """Return the smooth unit-sinusoid scenario."""

    signal = signals.sinusoid(samples_per_tau=samples_per_tau, cycles=cycles)
    tau = _required_characteristic_time(signal)
    return ReferenceScenario(
        name="sinusoidal",
        signal=signal,
        A_ref=1.0,
        tau=tau,
        w=tau,
        P_c=P_c,
        regime="periodic harmonic source",
        description="Unit sinusoid with the structural time equal to one period.",
    )


def damped(
    *, samples_per_tau: int = 64, cycles: int = 12, P_c: float = 1.0
) -> ReferenceScenario:
    """Return the passive underdamped-oscillator scenario."""

    signal = signals.damped_oscillator(samples_per_tau=samples_per_tau, cycles=cycles)
    tau = _required_characteristic_time(signal)
    return ReferenceScenario(
        name="damped",
        signal=signal,
        A_ref=1.0,
        tau=tau,
        w=tau,
        P_c=P_c,
        regime="passive / damped source",
        description="Underdamped passive oscillator with damping ratio 0.1.",
    )


def van_der_pol(*, samples_per_tau: int = 64, P_c: float = 1.0) -> ReferenceScenario:
    """Return the active self-sustained Van der Pol scenario."""

    signal = signals.van_der_pol(samples_per_tau=samples_per_tau)
    tau = _required_characteristic_time(signal)
    return ReferenceScenario(
        name="van_der_pol",
        signal=signal,
        A_ref=2.0,
        tau=tau,
        w=tau,
        P_c=P_c,
        regime="active / self-sustained source",
        description="Van der Pol mu=1 after fixed burn-in; tau=2*pi.",
    )


def unstable(
    *, samples_per_tau: int = 64, cycles: int = 10, P_c: float = 1.0
) -> ReferenceScenario:
    """Return the exponentially growing oscillator scenario."""

    signal = signals.unstable_oscillator(samples_per_tau=samples_per_tau, cycles=cycles)
    tau = _required_characteristic_time(signal)
    return ReferenceScenario(
        name="unstable",
        signal=signal,
        A_ref=1.0,
        tau=tau,
        w=tau,
        P_c=P_c,
        regime="unstable oscillating source",
        description="Negative-damping linear oscillator with fixed growth rate.",
    )


def stochastic(
    *, seed: int = 20260810, samples_per_tau: int = 50, P_c: float = 1.0
) -> ReferenceScenario:
    """Return the explicitly smoothed Ornstein-Uhlenbeck scenario."""

    signal = signals.smoothed_ornstein_uhlenbeck(
        seed=seed,
        samples_per_tau=samples_per_tau,
    )
    tau = _required_characteristic_time(signal)
    return ReferenceScenario(
        name="stochastic",
        signal=signal,
        A_ref=1.0,
        tau=tau,
        w=tau,
        P_c=P_c,
        regime="stochastic / regularized OU source",
        description="Seeded OU path followed by a fixed Gaussian low-pass filter.",
    )


def lorenz(*, samples_per_tau: int = 50, P_c: float = 1.0) -> ReferenceScenario:
    """Return the classical chaotic Lorenz-x scenario."""

    signal = signals.lorenz(samples_per_tau=samples_per_tau)
    tau = _required_characteristic_time(signal)
    return ReferenceScenario(
        name="lorenz",
        signal=signal,
        A_ref=20.0,
        tau=tau,
        w=tau,
        P_c=P_c,
        regime="chaotic / Lorenz source",
        description="Classical Lorenz parameters; observable u is the x coordinate.",
    )


__all__ = ["ReferenceScenario", "available", *_SCENARIOS]
