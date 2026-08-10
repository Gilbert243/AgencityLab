"""
Advanced correlations for Agencity.
"""

from __future__ import annotations
import numpy as np

EPS = 1e-12


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.std(a) * np.std(b)
    if denom < EPS:
        return 0.0

    return float(np.mean(a * b) / denom)


def correlation_b_components(b):
    """Correlation between real and imaginary parts."""
    b = np.asarray(b, complex)
    return _corr(np.real(b), np.imag(b))


def correlation_magnitude_phase(b):
    """Correlation between |b| and θ."""
    b = np.asarray(b, complex)

    mag = np.abs(b)
    theta = np.angle(b)

    return _corr(mag, theta)


def cross_correlation(a, b):
    """Cross-correlation (normalized)."""
    return _corr(a, b)


def full_correlation_summary(b, beta=None, J=None):
    """All correlations."""
    out = {
        "corr_real_imag": correlation_b_components(b),
        "corr_mag_phase": correlation_magnitude_phase(b),
    }

    if beta is not None:
        out["corr_b_beta"] = cross_correlation(np.abs(b), np.abs(beta))

    if J is not None:
        out["corr_b_J"] = cross_correlation(np.abs(b), J)

    return out