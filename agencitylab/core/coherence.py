"""
AgencityLab - core/coherence.py
================================

Structural coherence module for Agencity theory.

This module implements the fundamental mathematical tools related to:

    - structural orientation Θ
    - angular variance Σ_Θ
    - phase coherence
    - directional stability
    - circular statistics

No visualization or business logic should be placed here.

Author : AgencityLab
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================
# CONSTANTS
# ============================================================

EPSILON: float = 1e-12
TWO_PI: float = 2.0 * np.pi


# ============================================================
# ANGULAR UTILITIES
# ============================================================

def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """
    Wrap angles into the interval [-π, π].

    Parameters
    ----------
    theta : np.ndarray
        Array of angles in radians.

    Returns
    -------
    np.ndarray
        Wrapped angles.
    """
    return (theta + np.pi) % TWO_PI - np.pi


def angular_difference(
    theta1: np.ndarray,
    theta2: np.ndarray
) -> np.ndarray:
    """
    Compute the minimal angular difference.

    Parameters
    ----------
    theta1 : np.ndarray

    theta2 : np.ndarray

    Returns
    -------
    np.ndarray
        Angular difference in [-π, π].
    """
    return wrap_angle(theta1 - theta2)


# ============================================================
# STRUCTURAL ORIENTATION
# ============================================================

def compute_theta(
    M: np.ndarray,
    O: np.ndarray
) -> np.ndarray:
    """
    Compute structural orientation:

        Θ = atan2(O, M)

    Parameters
    ----------
    M : np.ndarray
        Structural memory component.

    O : np.ndarray
        Structural organization component.

    Returns
    -------
    np.ndarray
        Structural orientation in radians.
    """
    M = np.asarray(M, dtype=float)
    O = np.asarray(O, dtype=float)

    return np.arctan2(O, M)


# ============================================================
# CIRCULAR STATISTICS
# ============================================================

def circular_mean(theta: np.ndarray) -> float:
    """
    Compute the circular mean.

    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
    float
        Circular mean angle.
    """
    theta = np.asarray(theta, dtype=float)

    s = np.mean(np.sin(theta))
    c = np.mean(np.cos(theta))

    return np.arctan2(s, c)


def resultant_length(theta: np.ndarray) -> float:
    """
    Compute the mean resultant length.

    Interpretation
    --------------
    R ≈ 1:
        strong angular coherence

    R ≈ 0:
        angular dispersion

    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
    float
    """
    theta = np.asarray(theta, dtype=float)

    s = np.mean(np.sin(theta))
    c = np.mean(np.cos(theta))

    return np.sqrt(s**2 + c**2)


def circular_variance(theta: np.ndarray) -> float:
    """
    Compute circular variance.

    Definition
    ----------
        V = 1 - R

    where:
        R = resultant length

    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
    float
    """
    R = resultant_length(theta)

    return 1.0 - R


def circular_std(theta: np.ndarray) -> float:
    """
    Compute circular standard deviation.

    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
    float
    """
    R = resultant_length(theta)

    R = np.clip(R, EPSILON, 1.0)

    return np.sqrt(-2.0 * np.log(R))


# ============================================================
# ANGULAR VARIANCE Σ_Θ
# ============================================================

def angular_variance(
    theta: np.ndarray,
    window: Optional[int] = None
) -> np.ndarray:
    """
    Compute angular variance Σ_Θ.

    If window=None:
        return global variance.

    Otherwise:
        return sliding-window variance.

    Parameters
    ----------
    theta : np.ndarray
        Angular time series.

    window : int | None
        Sliding window size.

    Returns
    -------
    np.ndarray
    """
    theta = np.asarray(theta, dtype=float)

    if window is None:
        return np.array([
            circular_variance(theta)
        ])

    if window <= 1:
        raise ValueError(
            "window must be > 1"
        )

    n = len(theta)

    result = np.full(
        n,
        np.nan,
    )

    for i in range(window - 1, n):

        segment = theta[
            i - window + 1:
            i + 1
        ]

        result[i] = circular_variance(
            segment
        )

    return result


# ============================================================
# PHASE COHERENCE
# ============================================================

def phase_coherence(
    theta: np.ndarray
) -> float:
    """
    Compute phase coherence.

    Equivalent to the resultant length R.

    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
    float
    """
    return resultant_length(theta)


def directional_stability(
    theta: np.ndarray
) -> float:
    """
    Compute directional stability.

    Defined as:

        stability = 1 - circular_variance

    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
    float
    """
    return 1.0 - circular_variance(theta)


# ============================================================
# COHERENCE MATRIX
# ============================================================

def coherence_matrix(
    theta_series: np.ndarray
) -> np.ndarray:
    """
    Compute coherence matrix between
    multiple angular series.

    Parameters
    ----------
    theta_series : np.ndarray
        Shape:
            (n_series, n_time)

    Returns
    -------
    np.ndarray
        NxN coherence matrix.
    """
    theta_series = np.asarray(
        theta_series,
        dtype=float,
    )

    n_series = theta_series.shape[0]

    matrix = np.zeros(
        (n_series, n_series)
    )

    for i in range(n_series):

        for j in range(n_series):

            delta = angular_difference(
                theta_series[i],
                theta_series[j],
            )

            matrix[i, j] = np.mean(
                np.cos(delta)
            )

    return matrix


# ============================================================
# STRUCTURAL DIAGNOSTIC
# ============================================================

@dataclass
class CoherenceDiagnostic:
    """
    Fundamental structural coherence diagnostic.
    """

    theta_mean: float
    angular_variance: float
    circular_std: float
    phase_coherence: float
    directional_stability: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert diagnostic to dictionary.
        """
        return {
            "theta_mean": self.theta_mean,
            "angular_variance": self.angular_variance,
            "circular_std": self.circular_std,
            "phase_coherence": self.phase_coherence,
            "directional_stability": self.directional_stability,
        }


def coherence_diagnostic(
    M: np.ndarray,
    O: np.ndarray
) -> CoherenceDiagnostic:
    """
    Compute complete structural coherence diagnostic.

    Parameters
    ----------
    M : np.ndarray

    O : np.ndarray

    Returns
    -------
    CoherenceDiagnostic
    """
    theta = compute_theta(M, O)

    return CoherenceDiagnostic(
        theta_mean=circular_mean(theta),
        angular_variance=circular_variance(theta),
        circular_std=circular_std(theta),
        phase_coherence=phase_coherence(theta),
        directional_stability=directional_stability(theta),
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    t = np.linspace(
        0,
        20,
        2000,
    )

    M = np.cos(t)
    O = np.sin(t)

    theta = compute_theta(M, O)

    diag = coherence_diagnostic(M, O)

    print("\n=== COHERENCE DIAGNOSTIC ===")

    print(diag.to_dict())