"""
Detrending utilities for signals.
"""

from __future__ import annotations

import numpy as np


def detrend_signal(xi, u, method="linear"):
    
    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)

    if xi.ndim != 1:
        raise ValueError("xi must be 1D.")

    # 🔥 1D
    if u.ndim == 1:
        if method == "mean":
            return u - np.mean(u)

        if method == "linear":
            coeffs = np.polyfit(xi, u, 1)
            return u - np.polyval(coeffs, xi)

    # 🔥 ND
    elif u.ndim == 2:
        out = np.zeros_like(u)

        for i in range(u.shape[1]):
            col = u[:, i]

            if method == "mean":
                out[:, i] = col - np.mean(col)

            elif method == "linear":
                coeffs = np.polyfit(xi, col, 1)
                out[:, i] = col - np.polyval(coeffs, xi)

        return out

    raise ValueError("Unsupported dimension")