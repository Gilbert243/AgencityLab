"""Canonical observable agencity flux.

This module owns the scalar operator ``b = P_c * beta``. End-to-end
``u -> b`` orchestration is provided by :func:`agencitylab.compute_agencity`.
Diagnostics and interpretation belong to :mod:`agencitylab.analysis`.
"""

from __future__ import annotations

import numpy as np

from .validation import validate_nonnegative_scalar


def _validate_power_input(P_c, *, expected_shape):
    """Validate finite scalar or sampled ``P_c >= 0`` without altering values."""
    try:
        power = np.asarray(P_c, dtype=float)
    except Exception as exc:
        raise ValueError("P_c must be numeric") from exc

    if power.ndim == 0:
        return validate_nonnegative_scalar(float(power), name="P_c")
    if power.ndim != 1 or power.shape != expected_shape:
        raise ValueError("time-varying P_c must have the same one-dimensional shape as beta")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("P_c must contain only non-negative finite values")
    return power


def agencity(beta_signal, P_c=1.0, *, verbose=False):
    """Compute the canonical observable flux ``b(t) = P_c(t) * beta(t)`` exactly.

    ``P_c`` may be a finite non-negative scalar or sampled profile with the same
    shape as ``beta_signal``. In particular ``P_c = 0`` gives ``b = 0`` exactly.
    No signal-derived power, smoothing, saturation or epsilon replacement is
    introduced by this operator.
    """
    beta_signal = np.asarray(beta_signal, dtype=complex)
    if (
        beta_signal.ndim != 1
        or beta_signal.size == 0
        or not np.all(np.isfinite(beta_signal))
    ):
        raise ValueError("beta_signal must be a non-empty finite one-dimensional array")
    power = _validate_power_input(P_c, expected_shape=beta_signal.shape)
    b = power * beta_signal
    if verbose:
        print(f"[agencity] |b| mean={np.mean(np.abs(b)):.6g}")
    return b
