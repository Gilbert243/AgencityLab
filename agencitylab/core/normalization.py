"""Normalization utilities.

Canonical step:
    u -> u* = u / u_ref

The reference scale can be computed in several practical ways.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .safeguards import EPS, ensure_positive
from .validation import as_float_array, validate_signal


def center_signal(signal, *, axis: int = 0):
    """Remove the mean along the sample axis."""
    arr = as_float_array(signal)
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    return arr - mean


def compute_reference_scale(
    signal,
    *,
    method: str = "std",
    axis: int = 0,
    eps: float = EPS,
):
    """Compute a positive reference scale for normalization.

    Parameters
    ----------
    method:
        - 'std'  : standard deviation of the centered signal
        - 'range': peak-to-peak amplitude
        - 'mad'   : median absolute deviation (robust)
    """
    arr = as_float_array(signal)
    centered = center_signal(arr, axis=axis)

    if method == "std":
        scale = np.nanstd(centered, axis=axis, keepdims=True)
    elif method == "range":
        scale = np.nanmax(arr, axis=axis, keepdims=True) - np.nanmin(arr, axis=axis, keepdims=True)
    elif method == "mad":
        med = np.nanmedian(arr, axis=axis, keepdims=True)
        scale = np.nanmedian(np.abs(arr - med), axis=axis, keepdims=True)
    else:
        raise ValueError("Unknown reference scale method")

    scale = ensure_positive(scale, minimum=eps)
    return scale


def normalize_signal(
    signal,
    *,
    reference_scale=None,
    method: str = "std",
    center: bool = True,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize a signal into its reduced form u*.

    Returns
    -------
    u_star, u_ref
        The normalized signal and the reference scale used.
    """
    arr = validate_signal(signal)
    if center:
        arr = center_signal(arr, axis=axis)

    if reference_scale is None:
        reference_scale = compute_reference_scale(arr, method=method, axis=axis)
    else:
        reference_scale = ensure_positive(reference_scale)

    u_star = arr / reference_scale
    return u_star, reference_scale


def normalize_state(*args, **kwargs):
    """Alias kept for readability in external APIs."""
    return normalize_signal(*args, **kwargs)
