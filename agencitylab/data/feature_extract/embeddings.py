"""
Embedding-based feature construction (multi-dimensional, no information loss).
"""

from __future__ import annotations
from typing import Sequence, Literal

import numpy as np


def build_embedding_signal(
    vectors: Sequence[Sequence[float]],
    mode: Literal["raw", "norm", "mean", "sum", "pca"] = "raw",
    n_components: int | None = None,
) -> np.ndarray:
    """
    Convert embedding vectors into a signal representation.

    Modes
    -----
    - raw  : keep full vector (recommended)
    - norm : scalar norm (lossy)
    - mean : scalar mean (lossy)
    - sum  : scalar sum (lossy)
    - pca  : reduced but structured representation
    """

    arr = np.asarray(vectors, dtype=float)

    if arr.ndim != 2:
        raise ValueError("vectors must be 2D (n_samples, n_features)")

    normalized_mode = str(mode).lower().strip()

    # =========================
    # 🔥 1. FULL INFORMATION (RECOMMENDED)
    # =========================
    if normalized_mode == "raw":
        return arr  # shape (n, d)

    # =========================
    # ⚠️ LOSSY MODES
    # =========================
    if normalized_mode == "norm":
        return np.linalg.norm(arr, axis=1)

    if normalized_mode == "mean":
        return np.mean(arr, axis=1)

    if normalized_mode == "sum":
        return np.sum(arr, axis=1)

    # =========================
    # 🔬 PCA REDUCTION (SMART)
    # =========================
    if normalized_mode == "pca":
        # centrage
        X = arr - np.mean(arr, axis=0)

        # covariance
        C = np.cov(X, rowvar=False)

        # décomposition
        eigvals, eigvecs = np.linalg.eigh(C)

        # tri décroissant
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        if n_components is None:
            n_components = min(3, arr.shape[1])

        W = eigvecs[:, :n_components]

        return X @ W  # projection

    raise ValueError("Unknown mode.")