"""
Spectrum analysis (PRO VERSION, complex-aware).

Computes a frequency spectrum for a chosen component of b(t).
"""

from __future__ import annotations

import numpy as np

from agencitylab.core.safeguards import EPS


def _select_component(b, component: str = "magnitude"):
    b = np.asarray(b)
    if component == "magnitude":
        return np.abs(b).astype(float)
    if component == "real":
        return np.real(b).astype(float)
    if component == "imag":
        return np.imag(b).astype(float)
    if component == "phase":
        return np.unwrap(np.angle(b)).astype(float)
    if component == "complex":
        return b.astype(complex)
    raise ValueError("component must be one of: magnitude, real, imag, phase, complex")


def _safe_dt(xi):
    xi = np.asarray(xi, dtype=float)
    if xi.size < 2:
        return 1.0
    diffs = np.diff(xi)
    diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > EPS)]
    if diffs.size == 0:
        return 1.0
    return float(np.mean(np.abs(diffs)))


def agencity_spectrum(
    b,
    xi=None,
    *,
    component: str = "magnitude",
    detrend: bool = True,
    verbose: bool = False,
):
    """
    Frequency spectrum of the selected component.

    Parameters
    ----------
    b : array-like
        complex agencity signal
    xi : array-like or None
        coordinate axis
    component : str
        magnitude, real, imag, phase, complex
    detrend : bool
        subtract mean before FFT
    """
    x = _select_component(b, component=component)

    if xi is None:
        xi = np.arange(len(x), dtype=float)
    else:
        xi = np.asarray(xi, dtype=float)

    if x.size < 2:
        return {}

    if xi.size != x.size:
        raise ValueError("xi and b must have the same length")

    dt = _safe_dt(xi)

    signal = x - np.mean(x) if detrend and component != "complex" else x

    if component == "complex":
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), d=dt)
        mask = freq >= 0
        fft = fft[mask]
        freq = freq[mask]
        amp = np.abs(fft)
        power = amp ** 2
    else:
        fft = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(signal), d=dt)
        amp = np.abs(fft)
        power = amp ** 2

    if power.size == 0 or np.sum(power) <= EPS:
        return {
            "frequency": freq,
            "amplitude": amp,
            "power": power,
            "dominant_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "component": component,
        }

    peak_freq = float(freq[np.argmax(power)])
    spectral_centroid = float(np.sum(freq * power) / (np.sum(power) + EPS))

    p = power / (np.sum(power) + EPS)
    p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log(p + EPS)))

    if verbose:
        print("[spectrum] ---")
        print(f"[spectrum] component          : {component}")
        print(f"[spectrum] dt                 : {dt:.6g}")
        print(f"[spectrum] dominant_frequency : {peak_freq:.6g}")
        print(f"[spectrum] spectral_centroid   : {spectral_centroid:.6g}")
        print(f"[spectrum] spectral_entropy   : {spectral_entropy:.6g}")

    return {
        "frequency": freq,
        "amplitude": amp,
        "power": power,
        "dominant_frequency": peak_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "component": component,
    }


# Backward-compatible alias
frequency_spectrum = agencity_spectrum