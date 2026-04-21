"""
Embedding-based feature construction.

This module keeps the implementation lightweight and backend-agnostic.
Advanced vectorization can later be plugged in through optional dependencies.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def build_embedding_signal(vectors: Sequence[Sequence[float]], reduce: str = "norm") -> np.ndarray:
    """
    Convert a sequence of embedding vectors into a one-dimensional signal.

    Parameters
    ----------
    vectors:
        Sequence of vectors.
    reduce:
        Reduction strategy. Supported values:
        - norm
        - mean
        - sum
    """
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array-like structure.")

    reduce = reduce.lower().strip()

    if reduce == "norm":
        return np.linalg.norm(arr, axis=1)
    if reduce == "mean":
        return np.mean(arr, axis=1)
    if reduce == "sum":
        return np.sum(arr, axis=1)

    raise ValueError("Unknown reduction strategy.")
