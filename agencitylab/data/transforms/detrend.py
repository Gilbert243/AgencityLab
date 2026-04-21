"""
Detrending utilities for signals.
"""

from __future__ import annotations

import numpy as np


def detrend_signal(xi, u, method: str = "linear"):
    """
    Remove a slow trend from the signal values.

    Supported methods
    -----------------
    - linear
    - mean
    """
    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)

    if xi.ndim != 1 or u.ndim != 1:
        raise ValueError("xi and u must be one-dimensional.")

    method = method.lower().strip()

    if method == "mean":
        return u - np.mean(u)

    if method == "linear":
        coeffs = np.polyfit(xi, u, deg=1)
        trend = np.polyval(coeffs, xi)
        return u - trend

    raise ValueError("Unknown detrending method.")
