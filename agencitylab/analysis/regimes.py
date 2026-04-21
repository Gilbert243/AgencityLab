"""
Regime classification utilities for AgencityLab.

The classification is heuristic and intended for scientific exploration.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import agencity_mean, agencity_peak, agencity_variance


def classify_regime(b, epsilon: float = 1e-12) -> str:
    """
    Classify an Agencity trajectory into a qualitative regime.
    """
    b = np.asarray(b, dtype=float)

    if b.size == 0:
        return "unknown"

    peak = agencity_peak(b)
    variance = agencity_variance(b)
    mean = agencity_mean(b)

    if peak < epsilon:
        return "null"

    sign_changes = np.sum(np.diff(np.signbit(b - mean)) != 0)

    if variance < epsilon:
        return "stationary"

    if sign_changes == 0 and np.abs(mean) > 0.5 * peak:
        return "biased_active"

    if sign_changes > 0 and peak < 10.0 * np.std(b):
        return "oscillatory"

    if peak >= 10.0 * np.std(b):
        return "bursting"

    return "mixed"


def detect_regime_changes(b, window: int = 32, epsilon: float = 1e-12) -> List[int]:
    """
    Detect approximate indices where the regime changes.
    """
    b = np.asarray(b, dtype=float)
    if b.size < 2 * window:
        return []

    changes: List[int] = []
    prev = np.var(b[:window])

    for i in range(window, b.size - window):
        current = np.var(b[i - window : i + window])
        if abs(current - prev) > epsilon * max(1.0, abs(prev)):
            changes.append(i)
        prev = current

    return changes
