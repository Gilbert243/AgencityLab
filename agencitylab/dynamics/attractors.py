"""
Attractor classification utilities.

These diagnostics are heuristic and intended for exploratory analysis.
"""

from __future__ import annotations

import numpy as np


def detect_attractor_type(trajectory, epsilon: float = 1e-12) -> str:
    """
    Classify a trajectory in a qualitative way.

    Possible outputs:
    - fixed_point
    - periodic
    - quasi_periodic
    - chaotic_like
    - drifting
    - unknown
    """
    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.size == 0:
        return "unknown"

    if trajectory.ndim == 1:
        signal = trajectory
    else:
        signal = np.linalg.norm(trajectory, axis=1)

    amplitude = float(np.max(signal) - np.min(signal))
    if amplitude < epsilon:
        return "fixed_point"

    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if autocorr[0] < epsilon:
        return "unknown"

    autocorr = autocorr / autocorr[0]
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1

    if peaks.size == 0:
        if np.std(signal) < 1e-3 * amplitude:
            return "drifting"
        return "chaotic_like"

    if peaks.size == 1:
        return "periodic"

    if peaks.size >= 2:
        return "quasi_periodic"

    return "unknown"
