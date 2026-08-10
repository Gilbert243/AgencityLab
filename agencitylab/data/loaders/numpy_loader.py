"""
NumPy-oriented signal loader (multi-dimensional support).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def load_numpy_signal(data: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract xi and u from a NumPy-compatible object.

    Supports:
    - dict: {"xi": (n,), "u": (n,) or (n, d)}
    - array: shape (n, >=2)
        col 0 → xi
        col 1.. → u (possibly multi-dimensional)
    """

    # =========================
    # 🔥 DICT INPUT
    # =========================
    if isinstance(data, dict):
        if "xi" not in data or "u" not in data:
            raise ValueError('Dictionary must contain "xi" and "u".')

        xi = np.asarray(data["xi"], dtype=float)
        u = np.asarray(data["u"], dtype=float)

        if xi.ndim != 1:
            raise ValueError("xi must be 1D")

        if u.ndim == 1:
            return xi, u

        if u.ndim == 2:
            if u.shape[0] != xi.shape[0]:
                raise ValueError("xi and u must have same length")
            return xi, u

        raise ValueError("u must be 1D or 2D")

    # =========================
    # 🔥 ARRAY INPUT
    # =========================
    arr = np.asarray(data, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Array must be (n_samples, >=2)")

    xi = arr[:, 0]

    # 🔥 MULTI-D SUPPORT
    u = arr[:, 1:]

    # squeeze si scalaire
    if u.shape[1] == 1:
        u = u[:, 0]

    return xi, u