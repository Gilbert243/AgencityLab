"""
activity.py

Compute reduced activity A* from activation X*.

Definition (canonical):
A*(ξ*) = X*(ξ* + 1/2) - X*(ξ* - 1/2)
"""

import numpy as np


def activity(X_star: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Public canonical API for activity.
    """
    return compute_activity(X_star, window)


def activity_from_signal(X_star: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Alias for pipeline usage.
    """
    return compute_activity(X_star, window)


def compute_activity(X_star: np.ndarray, window: int = 1) -> np.ndarray:
    """
    Compute reduced activity A* using centered finite differences.
    """

    if window < 1:
        raise ValueError("window must be >= 1")

    X_star = np.asarray(X_star)

    if X_star.ndim != 1:
        raise ValueError("X_star must be a 1D array")

    n = len(X_star)
    A_star = np.zeros_like(X_star)

    for i in range(n):
        i_plus = min(i + window, n - 1)
        i_minus = max(i - window, 0)

        A_star[i] = X_star[i_plus] - X_star[i_minus]

    return A_star