"""
Windowing utilities for signals.
"""

from __future__ import annotations

import numpy as np


def apply_window(u, kind="hann"):
    
    u = np.asarray(u, dtype=float)
    n = u.shape[0]

    if kind == "hann":
        w = np.hanning(n)
    elif kind == "hamming":
        w = np.hamming(n)
    elif kind == "blackman":
        w = np.blackman(n)
    else:
        w = np.ones(n)

    # 🔥 1D
    if u.ndim == 1:
        return u * w

    # 🔥 ND (broadcast)
    elif u.ndim == 2:
        return u * w[:, None]

    raise ValueError("Unsupported dimension")