"""
Signal assembly helpers.

These utilities help create canonical u(ξ) sequences from intermediate
feature representations.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def build_signal_from_features(features: Sequence[Sequence[float]], reduce: str = "mean") -> np.ndarray:
    """
    Build a one-dimensional signal from feature vectors.
    """
    arr = np.asarray(features, dtype=float)
    if arr.ndim != 2:
        raise ValueError("features must be a 2D array-like structure.")

    reduce = reduce.lower().strip()

    if reduce == "mean":
        return np.mean(arr, axis=1)
    if reduce == "sum":
        return np.sum(arr, axis=1)
    if reduce == "norm":
        return np.linalg.norm(arr, axis=1)

    raise ValueError("Unknown feature reduction mode.")
