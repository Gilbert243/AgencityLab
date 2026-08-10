"""
Shannon-related helpers for AgencityLab.
"""

from __future__ import annotations
import numpy as np

EPS = 1e-12


def shannon_entropy(values, base: float = np.e, epsilon: float = EPS) -> float:
    """
    Shannon entropy H(X) from discrete values or weights.

    Accepts:
        - raw signal
        - histogram counts
        - probabilities
    """
    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return 0.0

    # if raw signal → convert to distribution
    if np.any(x < 0):
        x = np.abs(x)

    total = np.sum(x)
    if total <= epsilon:
        return 0.0

    p = x / total
    p = p[p > epsilon]

    log_base = np.log(base)
    return float(-np.sum(p * np.log(p)) / log_base)


def shannon_entropy_from_signal(signal, bins: int = 50) -> float:
    """
    Shannon entropy from continuous signal via histogram.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.size == 0:
        return 0.0

    hist, _ = np.histogram(signal, bins=bins)
    return shannon_entropy(hist)


def conditional_entropy(joint_distribution, axis: int = 0, base: float = np.e) -> float:
    """
    Conditional entropy H(X|Y) from joint distribution.
    """
    joint = np.asarray(joint_distribution, dtype=float)

    if joint.ndim != 2:
        raise ValueError("joint_distribution must be 2D")

    total = np.sum(joint)
    if total <= EPS:
        return 0.0

    pxy = joint / total

    if axis == 0:
        py = np.sum(pxy, axis=0, keepdims=True)
        cond = pxy / np.maximum(py, EPS)
    elif axis == 1:
        px = np.sum(pxy, axis=1, keepdims=True)
        cond = pxy / np.maximum(px, EPS)
    else:
        raise ValueError("axis must be 0 or 1")

    log_base = np.log(base)
    return float(-np.sum(pxy * np.log(np.maximum(cond, EPS))) / log_base)