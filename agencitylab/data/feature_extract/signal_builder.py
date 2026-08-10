"""
Signal assembly helpers.

These utilities help create canonical u(ξ) sequences from intermediate
feature representations.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def build_signal_from_features(features, mode="raw"):
    
    arr = np.asarray(features, dtype=float)

    if arr.ndim != 2:
        raise ValueError("features must be 2D")

    mode = mode.lower().strip()

    # 🔥 FULL
    if mode == "raw":
        return arr

    # ⚠️ réduction
    if mode == "mean":
        return np.mean(arr, axis=1)

    if mode == "norm":
        return np.linalg.norm(arr, axis=1)

    raise ValueError("Unknown mode")