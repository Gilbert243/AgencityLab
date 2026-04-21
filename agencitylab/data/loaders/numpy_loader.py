"""
NumPy-oriented signal loader.

The loader accepts a 2D array with two columns or a mapping containing
keys "xi" and "u".
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def load_numpy_signal(data: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract xi and u from a NumPy-compatible object.
    """
    if isinstance(data, dict):
        if "xi" not in data or "u" not in data:
            raise ValueError('Dictionary input must contain keys "xi" and "u".')
        return np.asarray(data["xi"], dtype=float), np.asarray(data["u"], dtype=float)

    arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Array input must have shape (n_samples, 2) or more.")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)
