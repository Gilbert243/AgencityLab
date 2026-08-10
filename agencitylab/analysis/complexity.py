"""
Dynamic complexity analysis.
"""

from __future__ import annotations
import numpy as np

EPS = 1e-12


def spectral_entropy(b):
    """Entropy of frequency spectrum."""
    b = np.asarray(b)
    fft = np.fft.rfft(np.abs(b))

    power = np.abs(fft)**2
    power = power / np.sum(power + EPS)

    power = power[power > EPS]

    return float(-np.sum(power * np.log(power)))


def lyapunov_like(b):
    """
    Approximate divergence rate.
    """
    b = np.asarray(b, complex)

    if len(b) < 2:
        return 0.0

    d = np.abs(np.diff(b))
    d = d[d > EPS]

    return float(np.mean(np.log(d)))


def fractal_dimension_proxy(b):
    """
    Rough fractal dimension proxy.
    """
    b = np.asarray(b, complex)

    n = len(b)
    if n < 10:
        return 1.0

    lengths = np.abs(np.diff(b))
    L = np.sum(lengths)

    return float(np.log(n) / (np.log(n) + np.log(L + EPS)))


def complexity_summary(b):
    return {
        "spectral_entropy": spectral_entropy(b),
        "lyapunov": lyapunov_like(b),
        "fractal_dimension": fractal_dimension_proxy(b),
    }