"""
Smoothing utilities for signals.
"""

from __future__ import annotations

import numpy as np


def smooth_signal(u, method="moving_average", window_size=5):
    
    u = np.asarray(u, dtype=float)

    if window_size == 1:
        return u.copy()

    def smooth_1d(x):
        if method == "moving_average":
            kernel = np.ones(window_size) / window_size
        else:
            center = (window_size - 1) / 2
            sigma = window_size / 6
            t = np.arange(window_size)
            kernel = np.exp(-0.5 * ((t - center)/sigma)**2)
            kernel /= kernel.sum()

        pad = window_size // 2
        padded = np.pad(x, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(x)]

    # 🔥 1D
    if u.ndim == 1:
        return smooth_1d(u)

    # 🔥 ND
    elif u.ndim == 2:
        return np.vstack([smooth_1d(u[:, i]) for i in range(u.shape[1])]).T

    raise ValueError("Unsupported dimension")