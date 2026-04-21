"""
Interpolation helpers for AgencityLab.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def interpolate_1d(x, y, x_new):
    """
    Linearly interpolate a 1D signal on a new grid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x_new.ndim != 1:
        raise ValueError("x, y and x_new must be one-dimensional.")
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    order = np.argsort(x)
    return x_new, np.interp(x_new, x[order], y[order])


def resample_uniform(x, y, num_points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a signal on a uniform grid.
    """
    x = np.asarray(x, dtype=float)
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")
    x_new = np.linspace(float(np.min(x)), float(np.max(x)), int(num_points))
    return interpolate_1d(x, y, x_new)
