"""
Numba-backed accelerations for AgencityLab.

This module is optional. It accelerates the heavy 1D loops when Numba is
available, while cleanly falling back to NumPy implementations otherwise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .numpy_backend import (
    apply_window_numpy,
    causal_moving_correlation_numpy,
    central_difference_numpy,
    normalize_numpy,
)


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
    try:
        return njit(cache=True)(func)
    except Exception:
        return func


def _central_difference_1d(values, step: float):
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    out = np.empty(n, dtype=np.float64)

    out[0] = (values[1] - values[0]) / step
    for i in range(1, n - 1):
        out[i] = (values[i + 1] - values[i - 1]) / (2.0 * step)
    out[n - 1] = (values[n - 1] - values[n - 2]) / step

    return out


def _causal_moving_correlation_1d(values, window: int = 1, epsilon: float = 1e-12):
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    result = np.zeros(n, dtype=np.float64)

    for i in range(2 * window - 1, n):
        a_start = i - 2 * window + 1
        a_end = i - window + 1
        b_start = i - window + 1
        b_end = i + 1

        a = values[a_start:a_end]
        b = values[b_start:b_end]

        mean_a = 0.0
        mean_b = 0.0
        for k in range(window):
            mean_a += a[k]
            mean_b += b[k]
        mean_a /= window
        mean_b /= window

        var_a = 0.0
        var_b = 0.0
        cov = 0.0
        for k in range(window):
            da = a[k] - mean_a
            db = b[k] - mean_b
            var_a += da * da
            var_b += db * db
            cov += da * db

        if var_a < epsilon or var_b < epsilon:
            result[i] = 0.0
        else:
            result[i] = cov / (np.sqrt(var_a * var_b) + epsilon)

    for i in range(0, 2 * window - 1):
        result[i] = 0.0

    return np.clip(result, -1.0, 1.0)


_central_difference_1d = compile_if_available(_central_difference_1d)
_causal_moving_correlation_1d = compile_if_available(_causal_moving_correlation_1d)


def normalize_numba(u, method: str = "zscore", epsilon: float = 1e-12):
    """
    Numba-facing normalization.

    For stability and simplicity, normalization still uses the NumPy backend,
    because it is already vectorized and rarely the main bottleneck.
    """
    return normalize_numpy(u, method=method, epsilon=epsilon, axis=None)


def central_difference_numba(values, step: float):
    """
    Numba-accelerated central difference for 1D signals.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError("values must contain at least two samples.")
    if step <= 0:
        raise ValueError("step must be positive.")
    return _central_difference_1d(values, step)


def apply_window_numba(values, kind: str = "hann"):
    """
    Windowing is already vectorized in NumPy, so keep the safe implementation.
    """
    return apply_window_numpy(values, kind=kind, axis=-1)


def causal_moving_correlation_numba(values, window: int = 1, epsilon: float = 1e-12):
    """
    Numba-accelerated causal moving correlation for 1D signals.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2 * window:
        raise ValueError("values must contain at least 2*window samples.")
    return _causal_moving_correlation_1d(values, window=window, epsilon=epsilon)