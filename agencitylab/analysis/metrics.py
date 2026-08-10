"""
Global metrics for AgencityLab (complex-aware).

This module provides scalar metrics computed from the complex agencity
observable b(t), using its magnitude by default.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from agencitylab.core.safeguards import EPS

Component = Literal["magnitude", "real", "imag", "phase"]


def agencity_components(b):
    """Return real part, imaginary part, magnitude and phase."""
    b = np.asarray(b)
    return np.real(b), np.imag(b), np.abs(b), np.angle(b)


def _select_component(b, component: Component = "magnitude"):
    """Select a scalar view of the complex signal."""
    b = np.asarray(b)
    if component == "magnitude":
        return np.abs(b)
    if component == "real":
        return np.real(b)
    if component == "imag":
        return np.imag(b)
    if component == "phase":
        return np.unwrap(np.angle(b))
    raise ValueError("component must be one of: magnitude, real, imag, phase")


def agencity_mean(b, component: Component = "magnitude") -> float:
    x = _select_component(b, component)
    return float(np.mean(x)) if x.size else 0.0


def agencity_variance(b, component: Component = "magnitude") -> float:
    x = _select_component(b, component)
    return float(np.var(x)) if x.size else 0.0


def agencity_peak(b) -> float:
    b = np.asarray(b)
    return float(np.max(np.abs(b))) if b.size else 0.0


def agencity_integral(xi, b, component: Component = "real"):
    """
    Discrete integral over the chosen component.

    For complex signals, default is the real part, because a complex
    trapezoidal integral is not always the most interpretable scalar.
    """
    xi = np.asarray(xi, dtype=float)
    x = _select_component(b, component)
    if xi.size < 2 or x.size < 2:
        return 0.0
    return float(np.trapz(x, xi))


def agencity_energy(b) -> float:
    """
    Energy-like quantity: sum(|b|^2).
    """
    b = np.asarray(b)
    return float(np.sum(np.abs(b) ** 2)) if b.size else 0.0


def agencity_power_mean(b) -> float:
    """
    Mean magnitude of the observable.
    """
    b = np.asarray(b)
    return float(np.mean(np.abs(b))) if b.size else 0.0


def shannon_entropy(b, bins: int = 50, component: Component = "magnitude") -> float:
    """
    Shannon entropy of a chosen scalar component.
    """
    x = _select_component(b, component)
    if x.size == 0:
        return 0.0

    counts, _ = np.histogram(x, bins=bins)
    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts.astype(float) / float(total)
    p = p[p > 0]
    if p.size == 0:
        return 0.0

    return float(-np.sum(p * np.log(p + EPS)))


def agencity_efficiency(b, P_c, epsilon: float = 1e-12):
    """
    Efficiency-like ratio |b| / |P_c|.
    Works with scalar or array-like P_c.
    """
    b = np.asarray(b)
    P_c = np.asarray(P_c)

    b_mag = np.abs(b)

    if P_c.shape != () and P_c.shape != b.shape:
        raise ValueError("b and P_c must have same shape or P_c must be scalar")

    return b_mag / np.maximum(np.abs(P_c), epsilon)


def global_efficiency(b, P_c) -> float:
    """
    Global efficiency of the system.
    """
    eta = agencity_efficiency(b, P_c)
    return float(np.mean(np.abs(eta))) if np.size(eta) else 0.0