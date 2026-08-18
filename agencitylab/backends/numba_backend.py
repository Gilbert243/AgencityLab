"""Optional Numba-backed primitives for AgencityLab experiments.

The canonical public pipeline remains the deterministic NumPy reference. Numba
is used here only where a measured, equivalent primitive exists. In particular,
CRM delegates to the shared O(N) NumPy implementation so that an optional JIT
path cannot silently introduce epsilon-modified physics.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np

from .numpy_backend import (
    WindowKind,
    apply_window_numpy,
    causal_moving_correlation_numpy,
    normalize_numpy,
)


def _require_numba():
    """Import Numba lazily."""
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
    """Compile a function with Numba when possible, otherwise return it unchanged."""
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


_central_difference_1d = compile_if_available(_central_difference_1d)


def normalize_numba(u, method: str = "zscore", epsilon: float = 1e-12):
    """Expose the optional diagnostic NumPy normalisation through this backend."""
    return normalize_numpy(u, method=method, epsilon=epsilon, axis=None)


def central_difference_numba(values, step: float):
    """Compute a Numba-accelerated central difference for one-dimensional data."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError("values must contain at least two samples.")
    if step <= 0:
        raise ValueError("step must be positive.")
    return _central_difference_1d(values, step)


def apply_window_numba(values, kind: str = "hann"):
    """Use NumPy for already-vectorised tapering."""
    return apply_window_numpy(values, kind=cast(WindowKind, kind), axis=-1)


def causal_moving_correlation_numba(values, window: int = 1, epsilon: float = 1e-12):
    """Return the shared canonical-convention O(N) CRM result.

    The historical Numba loop was O(N*w) and inserted ``epsilon`` into both the
    zero test and denominator. Delegation is deliberately faster for ordinary
    long signals and, more importantly, preserves the exact zero-variance rule.
    """
    return causal_moving_correlation_numpy(values, window=window, epsilon=epsilon)
