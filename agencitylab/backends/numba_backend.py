"""
Numba-backed accelerations for AgencityLab.

This module is optional. It mirrors a subset of the NumPy backend using
Numba when available.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _require_numba():
    """Import numba lazily."""
    try:
        from numba import njit  # type: ignore
        return njit
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Numba is not installed. Install AgencityLab with the numba extra."
        ) from exc


def has_numba() -> bool:
    """Return True if Numba is available."""
    try:
        _require_numba()
        return True
    except Exception:
        return False


def compile_if_available(func: Callable):
    """
    Compile a function with Numba if possible, otherwise return the original
    function unchanged.
    """
    try:
        njit = _require_numba()
    except Exception:
        return func
    return njit(cache=True)(func)


def normalize_numba(u, method: str = "zscore", epsilon: float = 1e-12):
    """
    Numba-accelerated normalization. Falls back to NumPy-like logic.
    """
    u = np.asarray(u, dtype=np.float64)

    if method == "zscore":
        mean = np.mean(u)
        std = np.std(u)
        if std < epsilon:
            return np.zeros_like(u)
        return (u - mean) / std

    if method == "minmax":
        u_min = np.min(u)
        u_max = np.max(u)
        span = u_max - u_min
        if span < epsilon:
            return np.zeros_like(u)
        return (u - u_min) / span

    if method == "centered":
        return u - np.mean(u)

    raise ValueError("Unknown normalization method.")
