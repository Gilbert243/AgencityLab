"""Reduced activation operator for AgencityLab."""

from __future__ import annotations

import numpy as np

from .validation import (
    is_exactly_constant,
    validate_axis,
    validate_positive_scalar,
    validate_signal,
)


def reduced_coordinate(axis, tau):
    """Return the canonical reduced time ``t* = t / tau``."""
    axis = validate_axis(axis)
    tau = validate_positive_scalar(tau, name="tau")
    return axis / tau


def _gradient(signal, axis):
    axis = validate_axis(axis, expected_length=signal.size)
    return np.gradient(signal, axis, edge_order=2 if signal.size > 2 else 1)


def compute_activation(
    u_star,
    axis,
    *,
    resolution_scale=None,
    replace_nan=False,
    clip=None,
    verbose=False,
):
    """Compute ``X* = d u* / d t*`` on the supplied reduced coordinate.

    ``axis`` must therefore be the reduced coordinate ``t*``. Physical
    coarse-graining or clipping must be performed outside the canonical operator.
    An exactly constant sampled state is recognized before finite differencing and
    returns exactly zero, following the canonical null-state convention.
    """
    if resolution_scale is not None or clip is not None or replace_nan:
        raise ValueError("canonical activation does not smooth, clip, or replace samples")

    u_star = validate_signal(u_star, name="u_star").ravel()
    if is_exactly_constant(u_star):
        X_star = np.zeros_like(u_star, dtype=float)
    else:
        X_star = _gradient(np.asarray(u_star, dtype=float), axis)

    if not np.all(np.isfinite(X_star)):
        raise ValueError("activation produced non-finite values")
    if verbose:
        print(f"[activation] mean={np.mean(X_star):.6g}")
    return np.asarray(X_star, dtype=float)


def activation(u_star, axis, **kwargs):
    """Canonical activation public API."""
    return compute_activation(u_star, axis, **kwargs)


def activation_from_signal(u_star, axis, **kwargs):
    """Pipeline readability alias."""
    return compute_activation(u_star, axis, **kwargs)
