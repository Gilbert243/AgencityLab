"""
Windowing utilities for signals.
"""

from __future__ import annotations

import numpy as np


def apply_window(u, kind: str = "hann"):
    """
    Apply a tapering window to the signal.

    Supported windows
    -----------------
    - hann
    - hamming
    - blackman
    - rectangular
    """
    u = np.asarray(u, dtype=float)

    if u.ndim != 1:
        raise ValueError("u must be one-dimensional.")

    n = u.shape[0]
    kind = kind.lower().strip()

    if kind == "hann":
        w = np.hanning(n)
    elif kind == "hamming":
        w = np.hamming(n)
    elif kind == "blackman":
        w = np.blackman(n)
    elif kind == "rectangular":
        w = np.ones(n, dtype=float)
    else:
        raise ValueError("Unknown window kind.")

    return u * w
