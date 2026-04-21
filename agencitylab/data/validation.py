"""
Validation helpers for the data layer.

The goal is to ensure that all preprocessing steps receive coherent inputs
and return a consistent canonical representation of a signal u(ξ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class SignalData:
    """
    Canonical representation of a one-dimensional signal.

    Parameters
    ----------
    xi:
        Coordinate array. It may represent time, iterations, distance,
        or any other ordering variable.
    u:
        Observed signal values sampled on xi.
    metadata:
        Optional free-form metadata.
    """
    xi: np.ndarray
    u: np.ndarray
    metadata: dict


def _as_1d_array(value: Any, name: str) -> np.ndarray:
    """Convert a sequence-like object into a one-dimensional NumPy array."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    return arr


def validate_signal_input(xi: Any, u: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate the canonical signal input pair (xi, u).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The validated coordinate and signal arrays.
    """
    xi_arr = _as_1d_array(xi, "xi")
    u_arr = _as_1d_array(u, "u")

    if xi_arr.shape[0] != u_arr.shape[0]:
        raise ValueError("xi and u must have the same length.")

    if not np.all(np.isfinite(xi_arr)):
        raise ValueError("xi contains non-finite values.")

    if not np.all(np.isfinite(u_arr)):
        raise ValueError("u contains non-finite values.")

    # Enforce monotonic ordering when possible.
    if np.any(np.diff(xi_arr) == 0):
        raise ValueError("xi must not contain repeated coordinates.")

    return xi_arr.astype(float), u_arr.astype(float)


def validate_input(data: Any) -> Any:
    """
    Validate a generic input object before loading or preprocessing.

    This function is intentionally permissive because the data layer is
    designed to accept a wide range of raw inputs.
    """
    if data is None:
        raise ValueError("Input data cannot be None.")
    return data
