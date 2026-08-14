"""Deterministic reference systems for scientific validation.

These generators do not define new Agencity physics. They provide reproducible
signals whose qualitative behaviour is stated in the theory documents so the
canonical implementation can be tested without tuning the equations to the
observed outputs.
"""

from __future__ import annotations

import numpy as np
from agencitylab.reference import scenarios
from agencitylab.reference.scenarios import ReferenceScenario


def rest_reference(*, samples_per_tau: int = 64) -> ReferenceScenario:
    return scenarios.rest(samples_per_tau=samples_per_tau)


def sine_reference(*, samples_per_tau: int = 64, cycles: int = 10) -> ReferenceScenario:
    return scenarios.sinusoidal(samples_per_tau=samples_per_tau, cycles=cycles)


def damped_reference(*, samples_per_tau: int = 64, cycles: int = 12) -> ReferenceScenario:
    return scenarios.damped(samples_per_tau=samples_per_tau, cycles=cycles)


def van_der_pol_reference(*, samples_per_tau: int = 64) -> ReferenceScenario:
    return scenarios.van_der_pol(samples_per_tau=samples_per_tau)


def unstable_reference(*, samples_per_tau: int = 64, cycles: int = 10) -> ReferenceScenario:
    return scenarios.unstable(samples_per_tau=samples_per_tau, cycles=cycles)


def filtered_ou_reference(*, samples_per_tau: int = 50) -> ReferenceScenario:
    return scenarios.stochastic(seed=20260810, samples_per_tau=samples_per_tau)


def lorenz_reference(*, samples_per_tau: int = 50) -> ReferenceScenario:
    return scenarios.lorenz(samples_per_tau=samples_per_tau)


def reference_suite() -> dict[str, ReferenceScenario]:
    """Return the fixed v0.4 reference battery."""
    return {
        "rest": rest_reference(),
        "sine": sine_reference(),
        "damped": damped_reference(),
        "van_der_pol": van_der_pol_reference(),
        "unstable": unstable_reference(),
        "filtered_ou": filtered_ou_reference(),
        "lorenz": lorenz_reference(),
    }


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
