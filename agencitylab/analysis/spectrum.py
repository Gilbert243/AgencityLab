"""
Spectrum analysis for AgencityLab.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def agencity_spectrum(b, xi=None, detrend: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute a simple one-sided amplitude spectrum of the observable b.
    """
    b = np.asarray(b, dtype=float)
    if xi is None:
        xi = np.arange(b.size, dtype=float)
    else:
        xi = np.asarray(xi, dtype=float)

    if b.size != xi.size:
        raise ValueError("xi and b must have the same length.")
    if b.size < 2:
        return {"frequency": np.asarray([]), "amplitude": np.asarray([])}

    signal = b.copy()
    if detrend:
        signal = signal - np.mean(signal)

    # Estimate a uniform step from the coordinate.
    dx = float(np.mean(np.diff(xi))) if xi.size > 1 else 1.0
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=dx)
    amp = np.abs(fft) / signal.size

    return {
        "frequency": freq,
        "amplitude": amp,
    }
