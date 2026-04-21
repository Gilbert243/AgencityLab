"""
Graph-to-signal utilities.

The base implementation works with simple adjacency matrices to avoid
forcing a graph library as a hard dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def graph_to_signal(graph: Any, mode: str = "degree_sequence") -> np.ndarray:
    """
    Convert a graph-like object into a one-dimensional signal.

    Accepted inputs
    ---------------
    - adjacency matrix
    - array-like square matrix
    """
    arr = np.asarray(graph, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("graph must be a square adjacency matrix or matrix-like object.")

    mode = mode.lower().strip()

    if mode == "degree_sequence":
        return np.sum(arr, axis=1)
    if mode == "column_sum":
        return np.sum(arr, axis=0)
    if mode == "laplacian_trace":
        degree = np.diag(np.sum(arr, axis=1))
        laplacian = degree - arr
        return np.asarray([float(np.trace(laplacian))], dtype=float)

    raise ValueError("Unknown graph-to-signal mode.")
