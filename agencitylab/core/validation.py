"""Validation helpers for the Agencity core."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .safeguards import EPS


def as_float_array(x, *, copy: bool = False):
    """Return a NumPy float array and optionally copy the data."""
    arr = np.asarray(x, dtype=float)
    return arr.copy() if copy else arr


def validate_signal(signal, *, name: str = "signal", min_length: int = 3):
    """Validate a one-dimensional signal or a sample-major array.

    The canonical core expects a sample axis and at least a small number of
    samples to compute derivatives and correlations safely.
    """
    arr = as_float_array(signal)
    if arr.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if arr.shape[0] < min_length:
        raise ValueError(f"{name} must contain at least {min_length} samples")
    return arr


def validate_axis(axis, *, expected_length: Optional[int] = None, name: str = "axis"):
    """Validate a coordinate array used as an evolution parameter."""
    arr = as_float_array(axis)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_length is not None and len(arr) != expected_length:
        raise ValueError(f"{name} length must match the signal length")
    if len(arr) < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def validate_window_size(window_size, *, name: str = "window_size"):
    """Validate a window size used in moving correlation calculations."""
    if window_size is None:
        raise ValueError(f"{name} cannot be None")
    try:
        value = float(window_size)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(value) or value <= EPS:
        raise ValueError(f"{name} must be strictly positive")
    return value
