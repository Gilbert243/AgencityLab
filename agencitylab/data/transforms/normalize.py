"""
Statistical normalization transforms (NON-CANONICAL).

⚠️ WARNING:
This module is for data preprocessing ONLY.

It must NOT be used in canonical Agencity computations,
which rely on fixed A_ref normalization:
    u* = u / A_ref
"""

from __future__ import annotations
import numpy as np


def normalize_signal(u, method="zscore", epsilon=1e-12):
    """
    Statistical normalization (experimental).

    Methods:
        - zscore
        - minmax
        - centered

    ⚠️ Not theory-compliant for Agencity core.
    """

    u = np.asarray(u, dtype=float)
    method = method.lower().strip()

    # =========================================================
    # 1D
    # =========================================================
    if u.ndim == 1:

        if method == "zscore":
            mean = np.mean(u)
            std = np.std(u)
            return (u - mean) / (std + epsilon)

        elif method == "minmax":
            return (u - u.min()) / (u.max() - u.min() + epsilon)

        elif method == "centered":
            return u - np.mean(u)

    # =========================================================
    # MULTI-D
    # =========================================================
    elif u.ndim == 2:

        if method == "zscore":
            mean = np.mean(u, axis=0, keepdims=True)
            std = np.std(u, axis=0, keepdims=True)
            return (u - mean) / (std + epsilon)

        elif method == "minmax":
            u_min = np.min(u, axis=0, keepdims=True)
            u_max = np.max(u, axis=0, keepdims=True)
            return (u - u_min) / (u_max - u_min + epsilon)

        elif method == "centered":
            return u - np.mean(u, axis=0, keepdims=True)

    raise ValueError(f"Unsupported normalization method '{method}' or input shape")