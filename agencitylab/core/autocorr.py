"""Autocorrelation utilities."""

from __future__ import annotations

import numpy as np

from .safeguards import EPS
from .validation import as_float_array


def autocorrelation(signal, *, demean: bool = True, normalize: bool = True):
    """Return the non-negative-lag autocorrelation of a 1D signal.

    The output is aligned with lag indices 0..N-1.
    """
    x = as_float_array(signal).ravel()
    if x.size < 2:
        raise ValueError("signal must contain at least two samples")

    if demean:
        x = x - np.nanmean(x)
    x = np.where(np.isfinite(x), x, 0.0)

    energy = np.dot(x, x)
    n = x.size
    if energy <= EPS:
        lags = np.arange(n, dtype=float)
        acf = np.zeros(n, dtype=float)
        acf[0] = 1.0
        return lags, acf

    corr = np.correlate(x, x, mode="full")
    acf = corr[n - 1 :]
    if normalize:
        acf = acf / max(acf[0], EPS)
    lags = np.arange(acf.size, dtype=float)
    return lags, acf


def normalized_autocorrelation(signal):
    """Convenience wrapper returning only the normalized autocorrelation."""
    return autocorrelation(signal, demean=True, normalize=True)[1]
