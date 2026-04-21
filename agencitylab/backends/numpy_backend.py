"""
NumPy backend for AgencityLab.

This module provides pure-NumPy implementations of the computational
primitives used by the core and dynamics layers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_numpy(u, method: str = "zscore", epsilon: float = 1e-12):
    """
    Normalize a one-dimensional signal.

    Supported methods:
    - zscore
    - minmax
    - centered
    """
    u = np.asarray(u, dtype=float)

    if u.ndim != 1:
        raise ValueError("u must be one-dimensional.")

    method = method.lower().strip()

    if method == "zscore":
        mean = float(np.mean(u))
        std = float(np.std(u))
        if std < epsilon:
            return np.zeros_like(u)
        return (u - mean) / std

    if method == "minmax":
        u_min = float(np.min(u))
        u_max = float(np.max(u))
        span = u_max - u_min
        if span < epsilon:
            return np.zeros_like(u)
        return (u - u_min) / span

    if method == "centered":
        return u - float(np.mean(u))

    raise ValueError("Unknown normalization method.")


def central_difference_numpy(values, step: float):
    """
    Compute the first derivative using central differences on a 1D grid.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if values.size < 2:
        raise ValueError("values must contain at least two samples.")

    derivative = np.zeros_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * step)
    derivative[0] = (values[1] - values[0]) / step
    derivative[-1] = (values[-1] - values[-2]) / step
    return derivative


def apply_window_numpy(values, kind: str = "hann"):
    """
    Apply a tapering window to a 1D signal.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    kind = kind.lower().strip()
    n = values.size

    if kind == "hann":
        window = np.hanning(n)
    elif kind == "hamming":
        window = np.hamming(n)
    elif kind == "blackman":
        window = np.blackman(n)
    elif kind == "rectangular":
        window = np.ones(n, dtype=float)
    else:
        raise ValueError("Unknown window kind.")

    return values * window


def causal_moving_correlation_numpy(values, window: int = 1, epsilon: float = 1e-12):
    """
    Compute a causal moving correlation between adjacent windows.

    This implementation returns an array of Pearson-like correlations in [-1, 1]
    using two successive windows of length 'window'.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if window < 1:
        raise ValueError("window must be >= 1.")
    if values.size < 2 * window:
        raise ValueError("values must contain at least 2*window samples.")

    result = np.zeros_like(values, dtype=float)

    for i in range(2 * window - 1, values.size):
        a = values[i - 2 * window + 1 : i - window + 1]
        b = values[i - window + 1 : i + 1]

        a0 = a - np.mean(a)
        b0 = b - np.mean(b)

        denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
        if denom < epsilon:
            result[i] = 0.0
        else:
            result[i] = float(np.dot(a0, b0) / denom)

    return np.clip(result, -1.0, 1.0)
