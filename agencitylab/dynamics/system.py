"""
Canonical dynamical system for AgencityLab.

The reference dynamics evolves the reduced variables:
X*, A*, M, O, and P_c.
The observable Agencity is then derived from beta and its variation.

This module is intentionally flexible: the constitutive laws can be
replaced by domain-specific closures without changing the overall structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import numpy as np


@dataclass(slots=True)
class AgencityState:
    """
    Canonical state vector for the dynamical layer.

    Attributes
    ----------
    X_star:
        Reduced activation.
    A_star:
        Reduced activity.
    M:
        Memory variable.
    O:
        Organization variable.
    P_c:
        Characteristic power.
    """
    X_star: float
    A_star: float
    M: float
    O: float
    P_c: float


def _tanh_clip(value: float) -> float:
    """Apply tanh after a safe float conversion."""
    return float(np.tanh(float(value)))


def beta_from_state(state: AgencityState) -> float:
    """
    Compute the canonical agencement beta from the state variables.
    """
    first_factor = _tanh_clip(state.X_star * (1.0 + state.A_star))
    second_factor = _tanh_clip(state.M + state.O)
    return float(first_factor * second_factor)


def agencity_rhs(
    xi_star: float,
    y: np.ndarray,
    constitutive_laws: Optional[Mapping[str, Callable[[float, np.ndarray], float]]] = None,
) -> np.ndarray:
    """
    Right-hand side of the canonical Agencity dynamical system.

    Parameters
    ----------
    xi_star:
        Reduced coordinate.
    y:
        State vector [X*, A*, M, O, P_c].
    constitutive_laws:
        Optional mapping with keys:
        - "A_star"
        - "M"
        - "O"
        - "P_c"

        Each callable must return the derivative for the corresponding
        variable.

    Returns
    -------
    np.ndarray
        Time derivative of the state vector.
    """
    y = np.asarray(y, dtype=float)
    if y.shape[0] != 5:
        raise ValueError("The state vector must have length 5: [X*, A*, M, O, P_c].")

    X_star, A_star, M, O, P_c = y
    state = AgencityState(X_star=X_star, A_star=A_star, M=M, O=O, P_c=P_c)

    if constitutive_laws is None:
        constitutive_laws = {}

    # Canonical closures used when no custom law is provided.
    default_A = lambda xi, yy: -0.5 * yy[0] + 0.25 * yy[1] + 0.1 * yy[2] - 0.1 * yy[3]
    default_M = lambda xi, yy: 0.5 * np.tanh(yy[1]) - 0.2 * yy[2]
    default_O = lambda xi, yy: 0.5 * np.tanh(yy[0]) - 0.2 * yy[3]
    default_P = lambda xi, yy: -0.1 * yy[4] + 0.05 * abs(beta_from_state(state))

    dX = A_star
    dA = constitutive_laws.get("A_star", default_A)(xi_star, y)
    dM = constitutive_laws.get("M", default_M)(xi_star, y)
    dO = constitutive_laws.get("O", default_O)(xi_star, y)
    dP = constitutive_laws.get("P_c", default_P)(xi_star, y)

    return np.asarray([dX, dA, dM, dO, dP], dtype=float)


def default_system_rhs(xi_star: float, y: np.ndarray) -> np.ndarray:
    """
    Canonical system RHS with default constitutive laws.
    """
    return agencity_rhs(xi_star, y, constitutive_laws=None)


def beta_and_b_from_trajectory(xi_star: np.ndarray, trajectory: np.ndarray, delta_star: float = 1.0):
    """
    Compute beta and the discrete Agencity observable b from a trajectory.

    Parameters
    ----------
    xi_star:
        Reduced coordinate samples.
    trajectory:
        Array of shape (n_samples, 5) with columns [X*, A*, M, O, P_c].
    delta_star:
        Reduced step used in the finite difference for b.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (beta, b_reduced)
    """
    xi_star = np.asarray(xi_star, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)

    if trajectory.ndim != 2 or trajectory.shape[1] != 5:
        raise ValueError("trajectory must have shape (n_samples, 5).")

    beta = np.zeros(trajectory.shape[0], dtype=float)
    for i in range(trajectory.shape[0]):
        state = AgencityState(*trajectory[i])
        beta[i] = beta_from_state(state)

    b_reduced = np.zeros_like(beta)
    if beta.size < 2:
        return beta, b_reduced

    b_reduced[1:] = (beta[1:] - beta[:-1]) / float(delta_star)
    b_reduced[0] = b_reduced[1]
    return beta, b_reduced
