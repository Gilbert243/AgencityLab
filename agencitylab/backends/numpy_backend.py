"""
NumPy backend for AgencityLab.

This module provides pure-NumPy implementations of the computational
primitives used by the core and optional acceleration layers.

It is the safe default backend and should never block users.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

WindowKind = Literal["hann", "hamming", "blackman", "rectangular"]


def normalize_numpy(u, method: str = "zscore", epsilon: float = 1e-12, axis=None):
    """
    Normalize an array.

    Supported methods:
    - zscore
    - minmax
    - centered

    Works on 1D or nD arrays.
    """
    u = np.asarray(u, dtype=float)
    method = str(method).lower().strip()

    if method == "zscore":
        mean = np.mean(u, axis=axis, keepdims=True)
        std = np.std(u, axis=axis, keepdims=True)
        return np.where(std < epsilon, np.zeros_like(u), (u - mean) / std)

    if method == "minmax":
        u_min = np.min(u, axis=axis, keepdims=True)
        u_max = np.max(u, axis=axis, keepdims=True)
        span = u_max - u_min
        return np.where(span < epsilon, np.zeros_like(u), (u - u_min) / span)

    if method == "centered":
        mean = np.mean(u, axis=axis, keepdims=True)
        return u - mean

    raise ValueError("Unknown normalization method.")


def central_difference_numpy(values, step: float, axis: int = -1):
    """
    Compute the first derivative using central differences.

    Supports nD arrays along the selected axis.
    """
    values = np.asarray(values, dtype=float)

    if step <= 0:
        raise ValueError("step must be positive.")
    if values.shape[axis] < 2:
        raise ValueError("values must contain at least two samples.")

    v = np.moveaxis(values, axis, -1)
    out = np.empty_like(v)

    out[..., 1:-1] = (v[..., 2:] - v[..., :-2]) / (2.0 * step)
    out[..., 0] = (v[..., 1] - v[..., 0]) / step
    out[..., -1] = (v[..., -1] - v[..., -2]) / step

    return np.moveaxis(out, -1, axis)


def apply_window_numpy(values, kind: WindowKind = "hann", axis: int = -1):
    """
    Apply a tapering window to an array along the selected axis.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[axis] < 1:
        raise ValueError("values must contain at least one sample.")

    kind = str(kind).lower().strip()
    n = values.shape[axis]

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

    shape = [1] * values.ndim
    shape[axis] = n
    return values * window.reshape(shape)


def causal_moving_correlation_numpy(values, window: int = 1, epsilon: float = 1e-12):
    """
    Compute a causal moving Pearson-like correlation on a 1D signal.

    The output is aligned with the most recent sample. Early positions are set to 0.
    """
    values = np.asarray(values, dtype=float).ravel()

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