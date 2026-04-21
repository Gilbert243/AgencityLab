"""
Shannon-related helpers for AgencityLab.

These functions are classical information-theoretic diagnostics and do not
claim to replace the Agencity observable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def shannon_entropy(values, base: float = np.e, epsilon: float = 1e-12) -> float:
    """
    Compute the Shannon entropy of a discrete distribution.

    Parameters
    ----------
    values:
        Non-negative weights or counts.
    base:
        Logarithm base. Default is e, which gives entropy in nats.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0

    total = float(np.sum(x))
    if total <= epsilon:
        return 0.0

    p = x / total
    p = p[p > epsilon]
    if p.size == 0:
        return 0.0

    log_base = np.log(base)
    return float(-np.sum(p * np.log(p)) / log_base)


def conditional_entropy(joint_distribution, axis: int = 0, base: float = np.e) -> float:
    """
    Compute a simple conditional entropy H(X|Y) from a joint distribution.
    """
    joint = np.asarray(joint_distribution, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint_distribution must be a 2D array.")
    pxy = joint / np.maximum(np.sum(joint), 1e-12)

    if axis == 0:
        py = np.sum(pxy, axis=0, keepdims=True)
        cond = np.divide(pxy, np.maximum(py, 1e-12))
    elif axis == 1:
        px = np.sum(pxy, axis=1, keepdims=True)
        cond = np.divide(pxy, np.maximum(px, 1e-12))
    else:
        raise ValueError("axis must be 0 or 1.")

    log_base = np.log(base)
    return float(-np.sum(pxy * np.log(np.maximum(cond, 1e-12))) / log_base)
