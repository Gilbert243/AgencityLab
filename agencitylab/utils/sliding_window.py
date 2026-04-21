"""
Sliding window helpers for AgencityLab.
"""

from __future__ import annotations

import numpy as np


def sliding_window_view_1d(values, window_size: int, step: int = 1):
    """
    Return a simple sliding-window view over a 1D array.

    The function avoids depending on newer NumPy APIs and works on old
    installations as well.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    if step < 1:
        raise ValueError("step must be >= 1.")
    if values.size < window_size:
        raise ValueError("window_size cannot exceed the length of values.")

    windows = []
    for start in range(0, values.size - window_size + 1, step):
        windows.append(values[start : start + window_size])
    return np.asarray(windows)
