"""
Graph-to-signal utilities.

The base implementation works with simple adjacency matrices to avoid
forcing a graph library as a hard dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def graph_to_signal(graph, mode="raw"):
    
    A = np.asarray(graph, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("graph must be square adjacency matrix")

    mode = mode.lower().strip()

    # 🔥 FULL STRUCTURE
    if mode == "raw":
        return A  # (n, n)

    # 🔥 NODE FEATURES
    if mode == "degree":
        return np.sum(A, axis=1, keepdims=True)

    if mode == "laplacian":
        D = np.diag(np.sum(A, axis=1))
        return D - A

    # 🔥 SPECTRAL (très puissant)
    if mode == "spectrum":
        eigvals = np.linalg.eigvalsh(A)
        return eigvals[:, None]

    raise ValueError("Unknown mode")