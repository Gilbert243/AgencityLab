"""
Stability indicators for AgencityLab trajectories.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def is_bounded_trajectory(trajectory, threshold: float = 1e6) -> bool:
    """
    Return True if the trajectory remains bounded below the given threshold.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.size == 0:
        return True
    return bool(np.all(np.abs(trajectory) <= threshold))


def lyapunov_like_indicator(trajectory, epsilon: float = 1e-12) -> float:
    """
    Compute a simple growth indicator based on consecutive separation ratios.

    This is not a full Lyapunov exponent, but a practical diagnostic used in
    the initial research version.
    """
    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.ndim == 1:
        diffs = np.abs(np.diff(trajectory))
    else:
        diffs = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    if diffs.size < 2:
        return 0.0

    ratios = (diffs[1:] + epsilon) / (diffs[:-1] + epsilon)
    return float(np.mean(np.log(ratios)))


def linear_stability_hint(jacobian_eigenvalues) -> str:
    """
    Provide a qualitative stability hint from eigenvalues.
    """
    eigenvalues = np.asarray(jacobian_eigenvalues, dtype=complex)
    if eigenvalues.size == 0:
        return "unknown"

    real_parts = np.real(eigenvalues)
    if np.all(real_parts < 0):
        return "stable"
    if np.any(real_parts > 0) and np.any(real_parts < 0):
        return "saddle-like"
    if np.allclose(real_parts, 0):
        return "neutral"
    if np.any(real_parts > 0):
        return "unstable"
    return "stable"
