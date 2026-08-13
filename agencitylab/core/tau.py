"""Characteristic-time utilities for AgencityLab."""

from __future__ import annotations

import numpy as np

from agencitylab.constants.characteristic_times import (
    resolve_characteristic_time,
    tau_context_from_metadata,
)
from .autocorr import autocorrelation
from .validation import validate_axis, validate_positive_scalar


def characteristic_time(*, tau="auto", system=None, domain=None, metadata=None, verbose=False):
    """Resolve canonical ``tau`` from explicit structural context.

    Metadata may carry an already specified ``characteristic_time``. No property of
    the observed signal is used by this resolver.
    """
    context = tau_context_from_metadata(metadata)
    if system is None:
        system = context.get("system")
    if domain is None:
        domain = context.get("domain")
    if (tau is None or str(tau).strip().lower() in {"auto", "canonical", "default"}) and context.get("tau") is not None:
        tau = context["tau"]

    value = resolve_characteristic_time(tau=tau, system=system, domain=domain, default=None)
    value = validate_positive_scalar(value, name="tau")
    if verbose:
        print(f"[tau] resolved tau={value}")
    return value


def _median_step(axis):
    axis = validate_axis(axis)
    return float(np.median(np.diff(axis)))


def _interpolate_threshold_crossing(lags, values, threshold):
    for i in range(1, len(values)):
        if values[i] <= threshold <= values[i - 1]:
            x0, x1 = lags[i - 1], lags[i]
            y0, y1 = values[i - 1], values[i]
            if y1 == y0:
                return float(x1)
            alpha = (threshold - y0) / (y1 - y0)
            return float(x0 + alpha * (x1 - x0))
    return None


def estimate_tau(
    activation_signal,
    *,
    axis=None,
    threshold: float = 0.5,
    fallback: str = "first_minimum",
    min_lag: int = 1,
    verbose: bool = False,
):
    """Experimental signal-derived estimate of ``tau``; not canonical physics."""
    if verbose:
        print("[tau] experimental signal-derived estimate")
    lags, acf = autocorrelation(activation_signal, demean=True, normalize=True)
    min_lag = max(1, int(min_lag))
    crossing = _interpolate_threshold_crossing(lags[min_lag:], acf[min_lag:], threshold)

    if crossing is None:
        if fallback == "first_minimum":
            idx = None
            for i in range(min_lag + 1, len(acf) - 1):
                if acf[i] <= acf[i - 1] and acf[i] <= acf[i + 1]:
                    idx = i
                    break
            crossing = float(idx if idx is not None else max(min_lag, len(acf) // 4))
        else:
            crossing = float(max(min_lag, len(acf) // 4))

    if axis is None:
        return float(crossing)
    return float(crossing * _median_step(axis))
