"""
Regime classification utilities for AgencityLab.

Complex-aware classification based on magnitude, variance, phase coherence,
and optional scaling exponent alpha.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from agencitylab.core.safeguards import EPS
from .metrics import agencity_mean, agencity_peak, agencity_variance


def _phase_stats(b):
    b = np.asarray(b)
    if b.size == 0:
        return 0.0, 0.0
    phase = np.unwrap(np.angle(b))
    return float(np.mean(phase)), float(np.std(phase))


def classify_regime(
    b,
    *,
    theta: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    epsilon: float = 1e-12,
    verbose: bool = False,
) -> str:
    """
    Classify an Agencity trajectory into a qualitative regime.

    Categories:
        - null
        - stationary
        - coherent
        - oscillatory
        - intermittent
        - bursting
        - dissipative
        - amplifying
        - scale-invariant
        - mixed
    """
    b = np.asarray(b)
    if b.size == 0:
        return "unknown"

    mag = np.abs(b)
    peak = agencity_peak(b)
    variance = agencity_variance(b, component="magnitude")
    mean_mag = agencity_mean(b, component="magnitude")
    std_mag = float(np.std(mag))

    if theta is not None:
        theta = np.asarray(theta, dtype=float)
        theta_std = float(np.std(theta)) if theta.size else 0.0
    else:
        _, theta_std = _phase_stats(b)

    if verbose:
        print("[regime] ---")
        print(f"[regime] mean(|b|) = {mean_mag:.6g}")
        print(f"[regime] peak(|b|) = {peak:.6g}")
        print(f"[regime] var(|b|)  = {variance:.6g}")
        print(f"[regime] std(theta)= {theta_std:.6g}")
        if alpha is not None:
            print(f"[regime] alpha     = {alpha:.6g}")

    if peak < epsilon:
        return "null"

    if variance < epsilon:
        return "stationary"

    if theta_std < 0.3 and mean_mag > 0.5 * peak:
        return "coherent"

    if peak >= 10.0 * (std_mag + epsilon):
        return "bursting"

    if theta_std >= 0.3 and peak < 10.0 * (std_mag + epsilon):
        return "oscillatory"

    if mean_mag > epsilon and variance > mean_mag:
        return "intermittent"

    if alpha is not None:
        if alpha > 0.1:
            return "amplifying"
        if alpha < -0.1:
            return "dissipative"
        return "scale-invariant"

    return "mixed"


def detect_regime_changes(
    b,
    *,
    window: int = 32,
    epsilon: float = 1e-12,
    component: str = "magnitude",
) -> List[int]:
    """
    Detect approximate indices where the regime changes.

    Uses rolling variance on the chosen scalar component.
    """
    b = np.asarray(b)
    if b.size < 2 * window:
        return []

    if component == "magnitude":
        x = np.abs(b)
    elif component == "real":
        x = np.real(b)
    elif component == "imag":
        x = np.imag(b)
    elif component == "phase":
        x = np.unwrap(np.angle(b))
    else:
        raise ValueError("component must be one of: magnitude, real, imag, phase")

    changes: List[int] = []
    prev = float(np.var(x[:window]))

    for i in range(window, x.size - window):
        current = float(np.var(x[i - window : i + window]))
        if abs(current - prev) > epsilon * max(1.0, abs(prev)):
            changes.append(i)
        prev = current

    return changes