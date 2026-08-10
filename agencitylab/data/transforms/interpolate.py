"""
Interpolation utilities for irregular coordinate grids.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def interpolate_signal(xi, u, new_xi, method="linear"):
    
    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)
    new_xi = np.asarray(new_xi, dtype=float)

    if xi.ndim != 1 or new_xi.ndim != 1:
        raise ValueError("xi and new_xi must be 1D.")

    if method != "linear":
        raise ValueError("Only linear interpolation supported.")

    order = np.argsort(xi)
    xi_sorted = xi[order]
    u_sorted = u[order]

    # 🔥 cas 1D
    if u.ndim == 1:
        new_u = np.interp(new_xi, xi_sorted, u_sorted)

    # 🔥 cas ND
    elif u.ndim == 2:
        new_u = np.vstack([
            np.interp(new_xi, xi_sorted, u_sorted[:, i])
            for i in range(u.shape[1])
        ]).T

    else:
        raise ValueError("Unsupported u dimension.")

    return new_xi, new_u