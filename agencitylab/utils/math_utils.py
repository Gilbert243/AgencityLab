"""
Mathematical helpers used across AgencityLab.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ensure_1d(values, name: str = "values") -> np.ndarray:
    """
    Convert an input into a one-dimensional NumPy array.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def nan_safe_mean(values, epsilon: float = 1e-12) -> float:
    """
    Return the mean of finite values, ignoring NaNs.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def clip_safely(values, lower: float = -1.0, upper: float = 1.0):
    """
    Clip values to a finite interval.
    """
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, lower, upper)


def moving_average(values, window_size: int = 5):
    """
    Compute a centered moving average of a 1D sequence.
    """
    arr = ensure_1d(values)
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    if window_size == 1:
        return arr.copy()
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    pad = window_size // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out[: arr.size]


def finite_difference(values, step: float):
    """
    Compute first-order finite differences using a centered stencil.
    """
    arr = ensure_1d(values)
    if step <= 0:
        raise ValueError("step must be positive.")
    if arr.size < 2:
        return np.zeros_like(arr)
    out = np.zeros_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * step)
    out[0] = (arr[1] - arr[0]) / step
    out[-1] = (arr[-1] - arr[-2]) / step
    return out
