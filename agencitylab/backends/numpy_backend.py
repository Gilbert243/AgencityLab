"""NumPy computational primitives for optional AgencityLab backend experiments.

The canonical public pipeline is implemented in :mod:`agencitylab.core` and
uses NumPy as its reference numerical engine. The helpers in this module remain
available for direct backend experiments, but they do not redefine canonical
normalisation, CRM, or any physical equation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from agencitylab.core.crm import _rolling_pearson

WindowKind = Literal["hann", "hamming", "blackman", "rectangular"]


def normalize_numpy(u, method: str = "zscore", epsilon: float = 1e-12, axis=None):
    """Apply a diagnostic array normalisation.

    These z-score/min-max helpers are optional preprocessing primitives. They
    are not the canonical ``u_star = u / A_ref`` normalisation.
    """
    u = np.asarray(u, dtype=float)
    method = str(method).lower().strip()

    if method == "zscore":
        mean = np.mean(u, axis=axis, keepdims=True)
        std = np.std(u, axis=axis, keepdims=True)
        return np.where(std < epsilon, np.zeros_like(u), (u - mean) / std)

    if method == "minmax":
        u_min = np.min(u, axis=axis, keepdims=True)
        u_max = np.max(u, axis=axis, keepdims=True)
        span = u_max - u_min
        return np.where(span < epsilon, np.zeros_like(u), (u - u_min) / span)

    if method == "centered":
        mean = np.mean(u, axis=axis, keepdims=True)
        return u - mean

    raise ValueError("Unknown normalization method.")


def central_difference_numpy(values, step: float, axis: int = -1):
    """Compute the first derivative using central differences."""
    values = np.asarray(values, dtype=float)

    if step <= 0:
        raise ValueError("step must be positive.")
    if values.shape[axis] < 2:
        raise ValueError("values must contain at least two samples.")

    v = np.moveaxis(values, axis, -1)
    out = np.empty_like(v)

    out[..., 1:-1] = (v[..., 2:] - v[..., :-2]) / (2.0 * step)
    out[..., 0] = (v[..., 1] - v[..., 0]) / step
    out[..., -1] = (v[..., -1] - v[..., -2]) / step

    return np.moveaxis(out, -1, axis)


def apply_window_numpy(values, kind: WindowKind = "hann", axis: int = -1):
    """Apply a tapering window to an array along the selected axis."""
    values = np.asarray(values, dtype=float)
    if values.shape[axis] < 1:
        raise ValueError("values must contain at least one sample.")

    key = str(kind).lower().strip()
    n = values.shape[axis]

    if key == "hann":
        window = np.hanning(n)
    elif key == "hamming":
        window = np.hamming(n)
    elif key == "blackman":
        window = np.blackman(n)
    elif key == "rectangular":
        window = np.ones(n, dtype=float)
    else:
        raise ValueError("Unknown window kind.")

    shape = [1] * values.ndim
    shape[axis] = n
    return values * window.reshape(shape)


def causal_moving_correlation_numpy(values, window: int = 1, epsilon: float = 1e-12):
    """Compute adjacent-window CRM with the canonical Pearson convention.

    ``epsilon`` is accepted for backward source compatibility but is not used in
    the Pearson denominator. Exactly zero empirical variance gives correlation
    zero; non-zero variance is never reclassified as zero by a threshold.
    """
    del epsilon
    values = np.asarray(values, dtype=float).ravel()

    if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
        raise ValueError("window must be an integer >= 1.")
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1.")
    if values.size < 2 * window:
        raise ValueError("values must contain at least 2*window samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite samples.")

    result, _ = _rolling_pearson(values, values, window, auto=True)
    return result
