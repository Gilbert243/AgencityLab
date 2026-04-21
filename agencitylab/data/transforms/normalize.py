"""
Normalization transforms for signals.
"""

from __future__ import annotations

import numpy as np


def normalize_signal(u, method: str = "zscore", epsilon: float = 1e-12):
    """
    Normalize a signal array.

    Supported methods
    -----------------
    - zscore: (u - mean) / std
    - minmax: scaled to [0, 1]
    - centered: u - mean
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
