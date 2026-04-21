"""
Smoothing utilities for signals.
"""

from __future__ import annotations

import numpy as np


def smooth_signal(u, method: str = "moving_average", window_size: int = 5):
    """
    Smooth a one-dimensional signal.

    Supported methods
    -----------------
    - moving_average
    - gaussian (approximated with a simple discrete kernel)
    """
    u = np.asarray(u, dtype=float)

    if u.ndim != 1:
        raise ValueError("u must be one-dimensional.")

    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    method = method.lower().strip()

    if window_size == 1:
        return u.copy()

    if method == "moving_average":
        kernel = np.ones(window_size, dtype=float) / float(window_size)

    elif method == "gaussian":
        center = (window_size - 1) / 2.0
        sigma = max(window_size / 6.0, 1e-12)
        x = np.arange(window_size, dtype=float)
        kernel = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        kernel /= np.sum(kernel)

    else:
        raise ValueError("Unknown smoothing method.")

    pad = window_size // 2
    padded = np.pad(u, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: u.shape[0]]
