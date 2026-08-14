"""Synthetic observable generators for scientific reference work.

The functions in this module generate observable samples only.  They never
compute, prescribe, or tune Agencity quantities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from agencitylab.fields.numerics import rk4_step
from agencitylab.models import AgencitySignal, ExperimentMetadata

_GENERATORS = (
    "constant",
    "sinusoid",
    "damped_oscillator",
    "van_der_pol",
    "unstable_oscillator",
    "white_noise",
    "ornstein_uhlenbeck",
    "smoothed_ornstein_uhlenbeck",
    "lorenz",
)


def available() -> tuple[str, ...]:
    """Return the public observable generator names."""

    return _GENERATORS


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return number


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _count(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _grid(tau: float, cycles: int, samples_per_tau: int) -> np.ndarray:
    characteristic_time = _positive(tau, "tau")
    n_cycles = _count(cycles, "cycles")
    resolution = _count(samples_per_tau, "samples_per_tau", minimum=4)
    dt = characteristic_time / resolution
    return np.arange(n_cycles * resolution + 1, dtype=float) * dt


def _metadata(
    *,
    title: str,
    description: str,
    system_type: str,
    tau: float,
    tags: list[str],
    extra: dict[str, Any] | None = None,
) -> ExperimentMetadata:
    return ExperimentMetadata(
        title=title,
        description=description,
        source="AgencityLab reference observable generator",
        domain="dynamical systems",
        system_type=system_type,
        observable_kind="synthetic scalar observable",
        characteristic_time=tau,
        tags=["reference", "synthetic", *tags],
        created_at="",
        extra={"scientific_status": "reference/test utility", **(extra or {})},
    )


def _signal(xi: np.ndarray, u: np.ndarray, metadata: ExperimentMetadata) -> AgencitySignal:
    if xi.shape != u.shape or xi.ndim != 1:
        raise RuntimeError("reference generator produced inconsistent array shapes")
    if not np.all(np.isfinite(xi)) or not np.all(np.isfinite(u)):
        raise RuntimeError("reference generator produced non-finite values")
    return AgencitySignal(xi=xi, u=u, metadata=metadata)


def constant(
    *, value: float = 2.5, tau: float = 1.0, cycles: int = 8, samples_per_tau: int = 64
) -> AgencitySignal:
    """Generate an exactly constant rest-state observable."""

    level = float(value)
    if not np.isfinite(level):
        raise ValueError("value must be finite")
    xi = _grid(tau, cycles, samples_per_tau)
    metadata = _metadata(
        title="Constant rest observable",
        description="Exact constant sampled rest state.",
        system_type="rest state",
        tau=float(tau),
        tags=["deterministic", "rest"],
        extra={"regularity": "C-infinity", "value": level},
    )
    return _signal(xi, np.full_like(xi, level), metadata)


def sinusoid(
    *,
    amplitude: float = 1.0,
    angular_frequency: float = 1.0,
    phase: float = 0.0,
    cycles: int = 10,
    samples_per_tau: int = 64,
) -> AgencitySignal:
    """Generate a smooth sinusoidal observable with ``tau`` equal to one period."""

    scale = _positive(amplitude, "amplitude")
    omega = _positive(angular_frequency, "angular_frequency")
    phase_value = float(phase)
    if not np.isfinite(phase_value):
        raise ValueError("phase must be finite")
    tau = 2.0 * np.pi / omega
    xi = _grid(tau, cycles, samples_per_tau)
    u = scale * np.sin(omega * xi + phase_value)
    metadata = _metadata(
        title="Sinusoidal observable",
        description="Smooth periodic scalar observable.",
        system_type="harmonic oscillator",
        tau=tau,
        tags=["deterministic", "periodic"],
        extra={
            "regularity": "C-infinity",
            "amplitude": scale,
            "angular_frequency": omega,
            "phase": phase_value,
            "expected_period": tau,
        },
    )
    return _signal(xi, u, metadata)


def damped_oscillator(
    *,
    damping_ratio: float = 0.1,
    natural_frequency: float = 1.0,
    amplitude: float = 1.0,
    cycles: int = 12,
    samples_per_tau: int = 64,
) -> AgencitySignal:
    """Generate the displacement of an underdamped passive oscillator."""

    zeta = _nonnegative(damping_ratio, "damping_ratio")
    if zeta >= 1.0:
        raise ValueError("damping_ratio must be < 1 for an underdamped oscillator")
    omega0 = _positive(natural_frequency, "natural_frequency")
    scale = _positive(amplitude, "amplitude")
    omega = omega0 * np.sqrt(1.0 - zeta**2)
    tau = 2.0 * np.pi / omega
    xi = _grid(tau, cycles, samples_per_tau)
    u = scale * np.exp(-zeta * omega0 * xi) * np.sin(omega * xi)
    metadata = _metadata(
        title="Damped oscillator observable",
        description="Underdamped passive oscillator displacement.",
        system_type="passive damped oscillator",
        tau=tau,
        tags=["deterministic", "passive", "damped"],
        extra={
            "regularity": "C-infinity",
            "damping_ratio": zeta,
            "natural_frequency": omega0,
            "expected_period": tau,
        },
    )
    return _signal(xi, u, metadata)


def _integrate_rk4(
    xi: np.ndarray,
    initial_state: np.ndarray,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    *,
    internal_substeps: int = 1,
) -> np.ndarray:
    substeps = _count(internal_substeps, "internal_substeps")
    states: np.ndarray = np.empty((xi.size, initial_state.size), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    for index, dt in enumerate(np.diff(xi), start=1):
        substep = float(dt) / substeps
        state = states[index - 1]
        start_time = float(xi[index - 1])
        for offset in range(substeps):
            state = rk4_step(rhs, start_time + offset * substep, state, substep)
        states[index] = state
    return states


def van_der_pol(
    *,
    mu: float = 1.0,
    initial_state: tuple[float, float] = (2.0, 0.0),
    burn_cycles: int = 10,
    output_cycles: int = 12,
    samples_per_tau: int = 64,
) -> AgencitySignal:
    """Generate a self-sustained Van der Pol displacement using NumPy RK4."""

    coefficient = _positive(mu, "mu")
    burn = _count(burn_cycles, "burn_cycles", minimum=0)
    output = _count(output_cycles, "output_cycles")
    tau = 2.0 * np.pi
    full_xi = _grid(tau, burn + output, samples_per_tau)
    state0 = np.asarray(initial_state, dtype=float)
    if state0.shape != (2,) or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must contain two finite values")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        x, velocity = state
        return np.array([velocity, coefficient * (1.0 - x * x) * velocity - x])

    states = _integrate_rk4(full_xi, state0, rhs)
    start = burn * _count(samples_per_tau, "samples_per_tau", minimum=4)
    xi = full_xi[start:] - full_xi[start]
    metadata = _metadata(
        title="Van der Pol observable",
        description="Self-sustained Van der Pol displacement after fixed burn-in.",
        system_type="self-sustained nonlinear oscillator",
        tau=tau,
        tags=["deterministic", "active", "limit-cycle"],
        extra={
            "regularity": "numerically smooth",
            "integrator": "classical RK4",
            "mu": coefficient,
            "burn_cycles": burn,
            "expected_period": 6.663286859,
        },
    )
    return _signal(xi, states[start:, 0], metadata)


def unstable_oscillator(
    *,
    growth_rate: float = 0.1,
    natural_frequency: float = 1.0,
    amplitude: float = 1.0,
    cycles: int = 10,
    samples_per_tau: int = 64,
) -> AgencitySignal:
    """Generate a linearly oscillating observable with exponential growth."""

    alpha = _nonnegative(growth_rate, "growth_rate")
    omega0 = _positive(natural_frequency, "natural_frequency")
    if alpha >= omega0:
        raise ValueError("growth_rate must be smaller than natural_frequency")
    scale = _positive(amplitude, "amplitude")
    omega = np.sqrt(omega0**2 - alpha**2)
    tau = 2.0 * np.pi / omega
    xi = _grid(tau, cycles, samples_per_tau)
    u = scale * np.exp(alpha * xi) * np.sin(omega * xi)
    metadata = _metadata(
        title="Unstable oscillator observable",
        description="Negative-damping linear oscillator displacement.",
        system_type="unstable linear oscillator",
        tau=tau,
        tags=["deterministic", "unstable"],
        extra={
            "regularity": "C-infinity",
            "growth_rate": alpha,
            "natural_frequency": omega0,
            "expected_period": tau,
        },
    )
    return _signal(xi, u, metadata)


def white_noise(
    *,
    seed: int = 42,
    scale: float = 1.0,
    tau: float = 1.0,
    cycles: int = 8,
    samples_per_tau: int = 64,
) -> AgencitySignal:
    """Generate raw discrete white noise.

    This output is reference data for preprocessing tests.  It is not presented
    as a ``C2`` observable suitable for the continuous canonical construction.
    """

    sigma = _positive(scale, "scale")
    xi = _grid(tau, cycles, samples_per_tau)
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, sigma, size=xi.size)
    metadata = _metadata(
        title="Raw white-noise samples",
        description="Raw discrete noise requiring an explicit regularization decision.",
        system_type="raw stochastic data",
        tau=float(tau),
        tags=["stochastic", "raw", "non-regular"],
        extra={
            "regularity": "raw discrete data; not C2",
            "seed": int(seed),
            "scale": sigma,
            "canonical_ready": False,
        },
    )
    return _signal(xi, u, metadata)


def ornstein_uhlenbeck(
    *,
    seed: int = 42,
    theta: float = 2.0,
    sigma: float = 0.35,
    mean: float = 0.0,
    tau: float = 1.0,
    burn_cycles: int = 8,
    output_cycles: int = 30,
    samples_per_tau: int = 50,
) -> AgencitySignal:
    """Generate raw Euler-Maruyama samples of an Ornstein-Uhlenbeck process."""

    rate = _positive(theta, "theta")
    diffusion = _positive(sigma, "sigma")
    long_mean = float(mean)
    if not np.isfinite(long_mean):
        raise ValueError("mean must be finite")
    characteristic_time = _positive(tau, "tau")
    burn = _count(burn_cycles, "burn_cycles", minimum=0)
    output = _count(output_cycles, "output_cycles")
    resolution = _count(samples_per_tau, "samples_per_tau", minimum=4)
    full_xi = _grid(characteristic_time, burn + output, resolution)
    dt = characteristic_time / resolution
    rng = np.random.default_rng(seed)
    process = np.empty_like(full_xi)
    process[0] = long_mean
    for index in range(1, process.size):
        previous = process[index - 1]
        process[index] = (
            previous + rate * (long_mean - previous) * dt + diffusion * np.sqrt(dt) * rng.normal()
        )
    start = burn * resolution
    xi = full_xi[start:] - full_xi[start]
    metadata = _metadata(
        title="Raw Ornstein-Uhlenbeck observable",
        description="Seeded OU sample path before regularization.",
        system_type="mean-reverting stochastic process",
        tau=characteristic_time,
        tags=["stochastic", "raw", "mean-reverting"],
        extra={
            "regularity": "continuous-time model with non-differentiable sample paths",
            "seed": int(seed),
            "theta": rate,
            "sigma": diffusion,
            "mean": long_mean,
            "canonical_ready": False,
        },
    )
    return _signal(xi, process[start:], metadata)


def _gaussian_smooth(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    width = _positive(sigma_samples, "smoothing_sigma_samples")
    radius = max(1, int(np.ceil(4.0 * width)))
    offsets: np.ndarray = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / width) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smoothed_ornstein_uhlenbeck(
    *,
    seed: int = 42,
    theta: float = 2.0,
    sigma: float = 0.35,
    mean: float = 0.0,
    tau: float = 1.0,
    burn_cycles: int = 8,
    output_cycles: int = 30,
    samples_per_tau: int = 50,
    smoothing_sigma_samples: float = 4.0,
) -> AgencitySignal:
    """Generate the regularized stochastic observable used by reference tests."""

    raw = ornstein_uhlenbeck(
        seed=seed,
        theta=theta,
        sigma=sigma,
        mean=mean,
        tau=tau,
        burn_cycles=burn_cycles,
        output_cycles=output_cycles,
        samples_per_tau=samples_per_tau,
    )
    u = _gaussian_smooth(raw.u, smoothing_sigma_samples)
    metadata = raw.metadata.with_updates(
        title="Smoothed Ornstein-Uhlenbeck observable",
        description="Seeded OU path followed by an explicit fixed Gaussian low-pass filter.",
        tags=["reference", "synthetic", "stochastic", "regularized", "mean-reverting"],
        extra={
            **raw.metadata.extra,
            "regularity": "discrete smooth reference observable",
            "canonical_ready": True,
            "regularization": "Gaussian convolution",
            "smoothing_sigma_samples": float(smoothing_sigma_samples),
        },
    )
    return _signal(raw.xi.copy(), u, metadata)


def lorenz(
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
    observable: str = "x",
    tau: float = 1.0,
    burn_cycles: int = 10,
    output_cycles: int = 30,
    samples_per_tau: int = 50,
) -> AgencitySignal:
    """Generate one coordinate of the classical Lorenz system using NumPy RK4."""

    sigma_value = _positive(sigma, "sigma")
    rho_value = _positive(rho, "rho")
    beta_value = _positive(beta, "beta")
    characteristic_time = _positive(tau, "tau")
    burn = _count(burn_cycles, "burn_cycles", minimum=0)
    output = _count(output_cycles, "output_cycles")
    resolution = _count(samples_per_tau, "samples_per_tau", minimum=4)
    components = {"x": 0, "y": 1, "z": 2}
    if observable not in components:
        raise ValueError("observable must be one of 'x', 'y', or 'z'")
    state0 = np.asarray(initial_state, dtype=float)
    if state0.shape != (3,) or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must contain three finite values")
    full_xi = _grid(characteristic_time, burn + output, resolution)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [sigma_value * (y - x), x * (rho_value - z) - y, x * y - beta_value * z]
        )

    states = _integrate_rk4(full_xi, state0, rhs, internal_substeps=4)
    start = burn * resolution
    xi = full_xi[start:] - full_xi[start]
    metadata = _metadata(
        title="Lorenz observable",
        description=f"Classical Lorenz {observable}-coordinate after fixed burn-in.",
        system_type="chaotic Lorenz system",
        tau=characteristic_time,
        tags=["deterministic", "chaotic"],
        extra={
            "regularity": "numerically smooth",
            "integrator": "classical RK4 with 4 internal substeps",
            "sigma": sigma_value,
            "rho": rho_value,
            "beta": beta_value,
            "observable_component": observable,
            "burn_cycles": burn,
        },
    )
    return _signal(xi, states[start:, components[observable]], metadata)


__all__ = list(_GENERATORS) + ["available"]
