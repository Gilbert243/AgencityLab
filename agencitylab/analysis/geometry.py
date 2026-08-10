"""
Geometric analysis of Agencity (complex plane).
"""

from __future__ import annotations
import numpy as np

EPS = 1e-12


def compute_angle(b, *, unwrap=True):
    """Return phase θ of complex signal."""
    b = np.asarray(b)
    theta = np.angle(b)
    return np.unwrap(theta) if unwrap else theta


def trajectory_length(b):
    """Length of trajectory in complex plane."""
    b = np.asarray(b, dtype=complex)
    db = np.diff(b)
    return float(np.sum(np.abs(db)))


def curvature(b):
    """
    Discrete curvature of trajectory.
    """
    b = np.asarray(b, dtype=complex)

    if len(b) < 3:
        return np.zeros_like(b, dtype=float)

    d1 = np.diff(b)
    d2 = np.diff(d1)

    k = np.zeros(len(b))
    k[1:-1] = np.abs(d2) / (np.abs(d1[:-1])**2 + EPS)

    return k


def radius(b):
    """Distance from origin."""
    return np.abs(np.asarray(b))


def geometric_summary(b):
    """Global geometric properties."""
    b = np.asarray(b)

    return {
        "radius_mean": float(np.mean(np.abs(b))),
        "radius_std": float(np.std(np.abs(b))),
        "trajectory_length": trajectory_length(b),
        "curvature_mean": float(np.mean(curvature(b))),
    }