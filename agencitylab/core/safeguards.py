"""Numerical safeguards used across the core layer.

These helpers keep the formulas stable in the presence of zero variance,
non-finite values, and overly large intermediate values.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def safe_divide(numerator, denominator, default=0.0):
    """Divide while protecting against near-zero denominators."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(numerator, denominator).shape, default, dtype=float)
    mask = np.abs(denominator) > EPS
    np.divide(numerator, denominator, out=out, where=mask)
    return out


def saturate(x, lower=-1.0, upper=1.0):
    """Clip values to a closed interval."""
    return np.clip(np.asarray(x, dtype=float), lower, upper)


def safe_tanh(x, clip=20.0):
    """Evaluate tanh after clipping the input to avoid overflow in upstream logic."""
    x = np.asarray(x, dtype=float)
    return np.tanh(np.clip(x, -clip, clip))


def replace_non_finite(x, default=0.0):
    """Replace NaN and infinite values by a finite fallback."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, default)


def ensure_positive(value, minimum=EPS):
    """Force a strictly positive scalar or array."""
    value = np.asarray(value, dtype=float)
    return np.maximum(np.abs(value), minimum)
