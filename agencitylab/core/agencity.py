"""Agencity observable b(t).

Canonical formula:
    b(t) = P_c(t) * [beta(t*) - beta(t* - Delta*)] / Delta*

This module provides a practical discrete approximation of that expression.
"""

from __future__ import annotations

import numpy as np

from .safeguards import EPS
from .validation import validate_axis, validate_window_size


def _broadcast_power(power, target_shape):
    power = np.asarray(power, dtype=float)
    if power.ndim == 0:
        return np.full(target_shape, float(power), dtype=float)
    return np.broadcast_to(power, target_shape).astype(float, copy=False)


def _effective_delta_star(delta_star, axis_star=None):
    if axis_star is None:
        return float(validate_window_size(delta_star))
    axis_star = validate_axis(axis_star)
    step = np.median(np.abs(np.diff(axis_star)))
    if not np.isfinite(step) or step <= EPS:
        step = 1.0
    delta_star = float(validate_window_size(delta_star))
    shift = max(1, int(round(delta_star / step)))
    return shift * step


def agencity(beta_signal, P_c=1.0, *, delta_star=1.0, axis_star=None, fill_value=np.nan):
    """Compute the observable agencity using a causal finite difference.

    Parameters
    ----------
    beta_signal:
        Structured agencement beta.
    P_c:
        Characteristic power, scalar or array-like.
    delta_star:
        Reduced time increment used by the finite difference.
    axis_star:
        Optional reduced coordinate. When provided, the actual step is inferred
        from the coordinate grid.
    """
    beta_signal = np.asarray(beta_signal, dtype=float).ravel()
    if beta_signal.size < 2:
        raise ValueError("beta_signal must contain at least two samples")

    delta_eff = _effective_delta_star(delta_star, axis_star=axis_star)
    if axis_star is None:
        shift = max(1, int(round(float(delta_star))))
    else:
        axis_star = validate_axis(axis_star, expected_length=beta_signal.size)
        step = np.median(np.abs(np.diff(axis_star)))
        if not np.isfinite(step) or step <= EPS:
            step = 1.0
        shift = max(1, int(round(delta_eff / step)))
        delta_eff = shift * step

    power = _broadcast_power(P_c, beta_signal.shape)
    out = np.full(beta_signal.shape, fill_value, dtype=float)
    out[shift:] = power[shift:] * (beta_signal[shift:] - beta_signal[:-shift]) / max(delta_eff, EPS)
    return out


def agencity_rate(beta_signal, *, delta_star=1.0, axis_star=None, fill_value=np.nan):
    """Return only the reduced derivative term Δbeta / Δ* ."""
    beta_signal = np.asarray(beta_signal, dtype=float).ravel()
    if beta_signal.size < 2:
        raise ValueError("beta_signal must contain at least two samples")
    delta_eff = _effective_delta_star(delta_star, axis_star=axis_star)
    if axis_star is None:
        shift = max(1, int(round(float(delta_star))))
    else:
        axis_star = validate_axis(axis_star, expected_length=beta_signal.size)
        step = np.median(np.abs(np.diff(axis_star)))
        if not np.isfinite(step) or step <= EPS:
            step = 1.0
        shift = max(1, int(round(delta_eff / step)))
        delta_eff = shift * step
    out = np.full(beta_signal.shape, fill_value, dtype=float)
    out[shift:] = (beta_signal[shift:] - beta_signal[:-shift]) / max(delta_eff, EPS)
    return out
