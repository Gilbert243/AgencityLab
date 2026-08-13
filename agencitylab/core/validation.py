"""Validation helpers for the canonical Agencity core."""

from __future__ import annotations

from typing import Optional

import numpy as np


def as_float_array(x, *, copy: bool = False):
    """Return a NumPy float array and optionally copy the data."""
    arr = np.asarray(x, dtype=float)
    return arr.copy() if copy else arr


def validate_signal(signal, *, name: str = "signal", min_length: int = 3):
    """Validate a finite sample-major signal."""
    arr = as_float_array(signal)

    if arr.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if arr.shape[0] < min_length:
        raise ValueError(f"{name} must contain at least {min_length} samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

    return arr


def is_exactly_constant(signal) -> bool:
    """Return True only when all sampled values are exactly identical.

    This is an exact discrete predicate, not an epsilon-based near-zero test. It is
    used by the canonical pipeline to recognize the postulated null/rest state
    before numerical differentiation or CRM evaluation.
    """
    arr = validate_signal(signal).ravel()
    return bool(np.all(arr == arr[0]))


def validate_axis(axis, *, expected_length: Optional[int] = None, name: str = "axis"):
    """Validate a finite, strictly increasing one-dimensional coordinate."""
    arr = as_float_array(axis)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_length is not None and len(arr) != expected_length:
        raise ValueError(f"{name} length must match the signal length")
    if len(arr) < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")

    return arr


def validate_window_size(window_size, *, name: str = "window_size"):
    """Validate a strictly positive finite physical window."""
    if window_size is None:
        raise ValueError(f"{name} cannot be None")
    try:
        value = float(window_size)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def validate_positive_scalar(value, *, name: str = "value"):
    """Validate a strictly positive finite scalar without epsilon substitution."""
    try:
        scalar = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc

    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return scalar


def validate_nonnegative_scalar(value, *, name: str = "value"):
    """Validate a finite scalar greater than or equal to zero exactly.

    This helper exists for quantities such as characteristic power ``P_c`` whose
    accepted physical domain includes zero. It does not replace zero by epsilon.
    """
    try:
        scalar = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc

    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return scalar
