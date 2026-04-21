"""
Global metrics for AgencityLab.

These metrics operate on the observable b and on the associated canonical
quantities stored in AgencityResult.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def agencity_mean(b) -> float:
    """Return the mean Agencity value."""
    b = np.asarray(b, dtype=float)
    return float(np.mean(b)) if b.size else 0.0


def agencity_variance(b) -> float:
    """Return the variance of Agencity."""
    b = np.asarray(b, dtype=float)
    return float(np.var(b)) if b.size else 0.0


def agencity_peak(b) -> float:
    """Return the maximum absolute Agencity value."""
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(b))) if b.size else 0.0


def agencity_integral(xi, b) -> float:
    """Return the numerical integral of b over xi."""
    xi = np.asarray(xi, dtype=float)
    b = np.asarray(b, dtype=float)
    if xi.size < 2:
        return 0.0
    return float(np.trapz(b, xi))


def agencity_efficiency(b, P_c, epsilon: float = 1e-12):
    """
    Return the reduced efficiency eta = b / P_c.
    """
    b = np.asarray(b, dtype=float)
    P_c = np.asarray(P_c, dtype=float)
    if b.shape != P_c.shape:
        raise ValueError("b and P_c must have the same shape.")
    return b / np.maximum(np.abs(P_c), epsilon)
