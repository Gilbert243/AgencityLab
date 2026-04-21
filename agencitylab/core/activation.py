"""
activation.py

Compute reduced activation X* from normalized signal u*.

Definition (canonical):
X*(ξ) = d(u*) / dξ
"""

import numpy as np


def activation(u_star: np.ndarray, dxi: float) -> np.ndarray:
    """
    Canonical activation function (public API).
    """
    return compute_activation(u_star, dxi)


def activation_from_signal(u_star: np.ndarray, dxi: float) -> np.ndarray:
    """
    Alias for clarity (pipeline usage).
    """
    return compute_activation(u_star, dxi)


def compute_activation(u_star: np.ndarray, dxi: float) -> np.ndarray:
    """
    Compute reduced activation X* = du*/dξ using finite differences.
    """

    if dxi <= 0:
        raise ValueError("dxi must be positive")

    u_star = np.asarray(u_star)

    if u_star.ndim != 1:
        raise ValueError("u_star must be a 1D array")

    X_star = np.zeros_like(u_star)

    # Central differences
    X_star[1:-1] = (u_star[2:] - u_star[:-2]) / (2 * dxi)

    # Boundaries
    X_star[0] = (u_star[1] - u_star[0]) / dxi
    X_star[-1] = (u_star[-1] - u_star[-2]) / dxi

    return X_star


def reduced_coordinate(xi: np.ndarray, tau: float) -> np.ndarray:
    """
    Compute reduced coordinate ξ* = ξ / τ
    """
    xi = np.asarray(xi, dtype=float)

    if tau <= 0:
        raise ValueError("tau must be positive")

    return xi / tau