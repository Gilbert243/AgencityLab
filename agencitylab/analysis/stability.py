"""
Stability helpers for AgencityLab.

Complex-aware summary on magnitude, phase, and component trends.
"""

from __future__ import annotations

import numpy as np


def stability_summary(b, *, verbose: bool = False):
    """
    Return a compact stability summary for a complex agencity signal.
    """
    b = np.asarray(b)

    if b.size < 2:
        return {}

    mag = np.abs(b)
    real = np.real(b)
    imag = np.imag(b)
    phase = np.unwrap(np.angle(b))

    x = np.arange(len(b), dtype=float)

    slope_mag = np.polyfit(x, mag, 1)[0] if len(b) >= 2 else 0.0
    slope_real = np.polyfit(x, real, 1)[0] if len(b) >= 2 else 0.0
    slope_imag = np.polyfit(x, imag, 1)[0] if len(b) >= 2 else 0.0

    variance = float(np.var(mag))
    amplitude = float(np.max(mag))

    osc_real = int(np.sum(np.diff(np.signbit(real - np.mean(real))) != 0))
    osc_imag = int(np.sum(np.diff(np.signbit(imag - np.mean(imag))) != 0))

    phase_rate = float(np.mean(np.abs(np.gradient(phase)))) if len(phase) >= 2 else 0.0
    phase_std = float(np.std(phase))

    stable = bool(abs(slope_mag) < 1e-6 and variance < 1.0)
    oscillatory = bool((osc_real + osc_imag) > len(b) * 0.1 or phase_rate > 0.1)

    out = {
        "trend_magnitude": float(slope_mag),
        "trend_real": float(slope_real),
        "trend_imag": float(slope_imag),
        "variance": variance,
        "amplitude": amplitude,
        "oscillations_real": osc_real,
        "oscillations_imag": osc_imag,
        "phase_std": phase_std,
        "phase_rate": phase_rate,
        "stable": stable,
        "oscillatory": oscillatory,
    }

    if verbose:
        print("[stability] ---")
        for k, v in out.items():
            print(f"[stability] {k}: {v}")

    return out