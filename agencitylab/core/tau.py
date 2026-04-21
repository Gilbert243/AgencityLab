"""Characteristic time estimation.

The canonical reference defines tau as the first lag where the normalized
autocorrelation of the activation drops below one half.
"""

from __future__ import annotations

import numpy as np

from .autocorr import autocorrelation
from .safeguards import EPS
from .validation import validate_axis


def _median_step(axis):
    axis = validate_axis(axis)
    diffs = np.diff(axis)
    diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > EPS)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(np.abs(diffs)))


def _interpolate_threshold_crossing(lags, values, threshold):
    for i in range(1, len(values)):
        if values[i] <= threshold <= values[i - 1]:
            x0, x1 = lags[i - 1], lags[i]
            y0, y1 = values[i - 1], values[i]
            if abs(y1 - y0) <= EPS:
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
):
    """Estimate the characteristic scale tau from an activation signal.

    Parameters
    ----------
    axis:
        Optional coordinate array. When provided, tau is returned in the same
        coordinate units as the axis.
    threshold:
        The normalized autocorrelation threshold.
    fallback:
        Strategy used when the threshold is never crossed.
    """
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
            if idx is None:
                idx = max(min_lag, len(acf) // 4)
            crossing = float(idx)
        elif fallback == "quarter_window":
            crossing = float(max(min_lag, len(acf) // 4))
        else:
            crossing = float(max(min_lag, len(acf) // 4))

    if axis is None:
        return float(crossing)

    step = _median_step(axis)
    return float(crossing * step)
