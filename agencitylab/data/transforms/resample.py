"""
Resampling utilities for signals.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .interpolate import interpolate_signal


def resample_signal(xi, u, num_points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample the signal on a uniform grid with the requested number of points.
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")

    xi = np.asarray(xi, dtype=float)
    new_xi = np.linspace(float(np.min(xi)), float(np.max(xi)), int(num_points))
    return interpolate_signal(xi, u, new_xi, method="linear")
