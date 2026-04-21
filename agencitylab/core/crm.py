"""Causal moving correlation (crm).

The canonical definition compares two adjacent windows of equal duration:
[t-1, t] and [t-2, t-1] in reduced coordinates.
"""

from __future__ import annotations

import numpy as np

from .safeguards import EPS
from .validation import as_float_array, validate_axis, validate_signal, validate_window_size


def _window_size_to_samples(window_size, axis=None):
    if isinstance(window_size, (int, np.integer)):
        return max(2, int(window_size))
    window_size = validate_window_size(window_size)
    if axis is None:
        return max(2, int(round(window_size)))
    axis = validate_axis(axis)
    step = np.median(np.abs(np.diff(axis)))
    if not np.isfinite(step) or step <= EPS:
        return max(2, int(round(window_size)))
    return max(2, int(round(window_size / step)))


def _pearson_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a - np.mean(a)
    b = b - np.mean(b)

    var_a = np.mean(a * a)
    var_b = np.mean(b * b)

    if var_a < EPS or var_b < EPS:
        return 0.0

    cov = np.mean(a * b)

    corr = cov / (np.sqrt(var_a * var_b) + EPS)

    # 🔥 CLAMP critique
    return float(np.clip(corr, -0.95, 0.95))

def causal_moving_correlation(signal, window_size=1.0, *, axis=None):
    """Compute the causal moving Pearson correlation on adjacent windows.

    The result is aligned with the end of the most recent window.
    Early positions where the second window is not available are filled with NaN.
    """
    x = validate_signal(signal).ravel()
    #n = _window_size_to_samples(window_size, axis=axis)
    n = max(5, _window_size_to_samples(window_size, axis=axis))
    if x.size < 2 * n:
        raise ValueError("signal is too short for the requested window size")

    #out = np.full(x.shape, np.nan, dtype=float)
    out = np.zeros_like(x, dtype=float)
    for end in range(2 * n - 1, x.size):
        #recent = x[end - n + 1 : end + 1]
        #previous = x[end - 2 * n + 1 : end - n + 1]
        
        lag = n // 2

        recent = x[end - n + 1 : end + 1]
        previous = x[end - n - lag + 1 : end - lag + 1]
        out[end] = _pearson_corr(recent, previous)
    return out


def crm_tau(signal, window_size=1.0, *, axis=None):
    """Alias for the canonical causal moving correlation operator."""
    return causal_moving_correlation(signal, window_size=window_size, axis=axis)
