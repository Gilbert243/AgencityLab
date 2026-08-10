"""Reduced activity operator for AgencityLab."""

from __future__ import annotations

import numpy as np

from .activation import _gradient
from .validation import is_exactly_constant, validate_signal


def compute_activity(
    X_star,
    axis,
    *,
    resolution_scale=None,
    replace_nan=False,
    clip=None,
    verbose=False,
):
    """Compute ``A* = d X* / d t*`` on the reduced coordinate ``t*``.

    An exactly constant sampled activation is recognized before finite
    differencing and returns exactly zero. No epsilon threshold is used.
    """
    if resolution_scale is not None or clip is not None or replace_nan:
        raise ValueError("canonical activity does not smooth, clip, or replace samples")

    X_star = validate_signal(X_star, name="X_star").ravel()
    if is_exactly_constant(X_star):
        A_star = np.zeros_like(X_star, dtype=float)
    else:
        A_star = _gradient(np.asarray(X_star, dtype=float), axis)

    if not np.all(np.isfinite(A_star)):
        raise ValueError("activity produced non-finite values")
    if verbose:
        print(f"[activity] mean={np.mean(A_star):.6g}")
    return np.asarray(A_star, dtype=float)


def activity(X_star, axis, **kwargs):
    """Canonical activity public API."""
    return compute_activity(X_star, axis, **kwargs)


def activity_from_signal(X_star, axis, **kwargs):
    """Pipeline readability alias."""
    return compute_activity(X_star, axis, **kwargs)
