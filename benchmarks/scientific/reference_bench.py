"""Deterministic reference systems for scientific validation.

These generators do not define new Agencity physics. They provide reproducible
signals whose qualitative behaviour is stated in the theory documents so the
canonical implementation can be tested without tuning the equations to the
observed outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d

from agencitylab import compute_agencity


@dataclass(frozen=True)
class ReferenceSignal:
    """One fixed physical/numerical context for a validation signal."""

    name: str
    xi: np.ndarray
    u: np.ndarray
    A_ref: float
    tau: float
    P_c: float = 1.0
    expected_period: float | None = None
    note: str = ""

    def compute(self):
        """Evaluate the public canonical API on this reference signal."""
        return compute_agencity(
            u=self.u,
            xi=self.xi,
            A_ref=self.A_ref,
            tau=self.tau,
            P_c=self.P_c,
        )


def _grid(tau: float, cycles: int, samples_per_tau: int) -> np.ndarray:
    dt = float(tau) / int(samples_per_tau)
    return np.arange(cycles * samples_per_tau + 1, dtype=float) * dt


def rest_reference(*, samples_per_tau: int = 64) -> ReferenceSignal:
    tau = 1.0
    xi = _grid(tau, 8, samples_per_tau)
    return ReferenceSignal(
        name="rest",
        xi=xi,
        u=np.full_like(xi, 2.5),
        A_ref=1.0,
        tau=tau,
        note="Exact constant sampled rest state.",
    )


def sine_reference(*, samples_per_tau: int = 64, cycles: int = 10) -> ReferenceSignal:
    tau = 2.0 * np.pi
    xi = _grid(tau, cycles, samples_per_tau)
    return ReferenceSignal(
        name="sine",
        xi=xi,
        u=np.sin(xi),
        A_ref=1.0,
        tau=tau,
        expected_period=tau,
        note="Unit sinusoid with the structural time equal to one period.",
    )


def damped_reference(*, samples_per_tau: int = 64, cycles: int = 12) -> ReferenceSignal:
    zeta = 0.1
    omega0 = 1.0
    omega = omega0 * np.sqrt(1.0 - zeta**2)
    tau = 2.0 * np.pi / omega
    xi = _grid(tau, cycles, samples_per_tau)
    u = np.exp(-zeta * xi) * np.sin(omega * xi)
    return ReferenceSignal(
        name="damped",
        xi=xi,
        u=u,
        A_ref=1.0,
        tau=tau,
        expected_period=tau,
        note="Underdamped passive oscillator with zeta=0.1 and omega0=1.",
    )


def van_der_pol_reference(*, samples_per_tau: int = 64) -> ReferenceSignal:
    mu = 1.0
    tau = 2.0 * np.pi
    burn_cycles = 10
    output_cycles = 12
    full_xi = _grid(tau, burn_cycles + output_cycles, samples_per_tau)

    def rhs(_t, state):
        x, velocity = state
        return (velocity, mu * (1.0 - x * x) * velocity - x)

    solution = solve_ivp(
        rhs,
        (float(full_xi[0]), float(full_xi[-1])),
        (2.0, 0.0),
        t_eval=full_xi,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Van der Pol reference integration failed: {solution.message}")

    start = burn_cycles * samples_per_tau
    xi = full_xi[start:] - full_xi[start]
    u = solution.y[0, start:]
    return ReferenceSignal(
        name="van_der_pol",
        xi=xi,
        u=u,
        A_ref=2.0,
        tau=tau,
        expected_period=6.663286859,
        note="Van der Pol mu=1 after a fixed burn-in; tau=2*pi follows the reference analysis.",
    )


def unstable_reference(*, samples_per_tau: int = 64, cycles: int = 10) -> ReferenceSignal:
    alpha = 0.1
    omega0 = 1.0
    omega = np.sqrt(omega0**2 - alpha**2)
    tau = 2.0 * np.pi / omega
    xi = _grid(tau, cycles, samples_per_tau)
    u = np.exp(alpha * xi) * np.sin(omega * xi)
    return ReferenceSignal(
        name="unstable",
        xi=xi,
        u=u,
        A_ref=1.0,
        tau=tau,
        expected_period=tau,
        note="Negative-damping linear oscillator with alpha=0.1 and omega0=1.",
    )


def filtered_ou_reference(*, samples_per_tau: int = 50) -> ReferenceSignal:
    tau = 1.0
    burn_cycles = 8
    output_cycles = 30
    full_xi = _grid(tau, burn_cycles + output_cycles, samples_per_tau)
    dt = tau / samples_per_tau
    theta = 2.0
    sigma = 0.35
    rng = np.random.default_rng(20260810)
    process = np.zeros_like(full_xi)
    for index in range(1, process.size):
        process[index] = (
            process[index - 1]
            - theta * process[index - 1] * dt
            + sigma * np.sqrt(dt) * rng.normal()
        )

    filtered = gaussian_filter1d(process, sigma=4.0, mode="nearest")
    start = burn_cycles * samples_per_tau
    xi = full_xi[start:] - full_xi[start]
    u = filtered[start:]
    return ReferenceSignal(
        name="filtered_ou",
        xi=xi,
        u=u,
        A_ref=1.0,
        tau=tau,
        note="Seeded Ornstein-Uhlenbeck process followed by a fixed Gaussian low-pass filter.",
    )


def lorenz_reference(*, samples_per_tau: int = 50) -> ReferenceSignal:
    tau = 1.0
    burn_cycles = 10
    output_cycles = 30
    full_xi = _grid(tau, burn_cycles + output_cycles, samples_per_tau)
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def rhs(_t, state):
        x, y, z = state
        return (
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        )

    solution = solve_ivp(
        rhs,
        (float(full_xi[0]), float(full_xi[-1])),
        (1.0, 1.0, 1.0),
        t_eval=full_xi,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Lorenz reference integration failed: {solution.message}")

    start = burn_cycles * samples_per_tau
    xi = full_xi[start:] - full_xi[start]
    u = solution.y[0, start:]
    return ReferenceSignal(
        name="lorenz",
        xi=xi,
        u=u,
        A_ref=20.0,
        tau=tau,
        note="Classical Lorenz parameters sigma=10, rho=28, beta=8/3; observable u=x.",
    )


def reference_suite() -> dict[str, ReferenceSignal]:
    """Return the fixed v0.4 reference battery."""
    cases = (
        rest_reference(),
        sine_reference(),
        damped_reference(),
        van_der_pol_reference(),
        unstable_reference(),
        filtered_ou_reference(),
        lorenz_reference(),
    )
    return {case.name: case for case in cases}


def structural_mask(result) -> np.ndarray:
    """Mask samples with complete finite-record CRM history.

    The implementation initializes finite-record CRM values before two complete
    windows to zero because the unavailable prehistory is unknown. Scientific
    regime comparisons therefore use t >= t0 + 2*tau. This is a numerical
    boundary convention, not a modification of the theory.
    """
    return result.xi >= result.xi[0] + 2.0 * result.tau


def circular_variance(result) -> float:
    """Circular variance of canonical structural orientation on valid samples."""
    mask = structural_mask(result) & (result.S > 0.0)
    if not np.any(mask):
        return float("nan")
    directions = np.exp(1j * result.theta[mask])
    return float(1.0 - np.abs(np.mean(directions)))


def periodic_relative_error(result, period: float, *, tail_fraction: float = 0.5) -> float:
    """Relative RMS mismatch after shifting beta by one candidate period.

    A two-derivative finite-difference pipeline can propagate endpoint stencils a
    few samples inward. The periodicity metric therefore compares only interior
    samples; endpoint accuracy is covered by the separate convergence test.
    """
    step = float(np.median(np.diff(result.xi)))
    shift = max(1, int(round(float(period) / step)))
    edge = 3
    start = max(
        edge,
        int(round(tail_fraction * len(result))),
        int(np.searchsorted(result.xi, result.xi[0] + 2.0 * result.tau)),
    )
    stop = len(result) - shift - edge
    if stop <= start:
        raise ValueError("not enough interior samples for periodicity comparison")

    left = result.beta[start:stop]
    right = result.beta[start + shift : stop + shift]
    rms = float(np.sqrt(np.mean(np.abs(left) ** 2)))
    mismatch = float(np.sqrt(np.mean(np.abs(left - right) ** 2)))
    return mismatch / max(rms, np.finfo(float).tiny)


def benchmark_summary() -> dict[str, dict[str, float]]:
    """Compute compact deterministic metrics for the complete reference suite."""
    summary = {}
    for name, case in reference_suite().items():
        result = case.compute()
        mask = structural_mask(result)
        summary[name] = {
            "mean_abs_beta": float(np.mean(np.abs(result.beta[mask]))),
            "mean_J": float(np.mean(result.J[mask])),
            "mean_S": float(np.mean(result.S[mask])),
            "orientation_circular_variance": circular_variance(result),
        }
    return summary


if __name__ == "__main__":
    for case_name, metrics in benchmark_summary().items():
        print(case_name)
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.8g}")
