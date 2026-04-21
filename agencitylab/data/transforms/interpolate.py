"""
Interpolation utilities for irregular coordinate grids.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def interpolate_signal(xi, u, new_xi, method: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a signal onto a new coordinate grid.

    Only linear interpolation is provided in the base layer to keep the
    installation light.
    """
    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)
    new_xi = np.asarray(new_xi, dtype=float)

    if xi.ndim != 1 or u.ndim != 1 or new_xi.ndim != 1:
        raise ValueError("xi, u, and new_xi must be one-dimensional.")

    if method.lower().strip() != "linear":
        raise ValueError("Only linear interpolation is available in the base layer.")

    order = np.argsort(xi)
    xi_sorted = xi[order]
    u_sorted = u[order]
    new_u = np.interp(new_xi, xi_sorted, u_sorted)
    return new_xi, new_u
