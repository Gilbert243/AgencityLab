"""
Advanced diagnostic helpers for AgencityLab.

Complex-aware event, transition, and regime summaries.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .regimes import classify_regime


def detect_events(b, threshold: float = 3.0, *, component: str = "magnitude"):
    """
    Detect strong outliers using z-score on the chosen component.
    """
    b = np.asarray(b)

    if b.size == 0:
        return np.asarray([], dtype=int)

    if component == "magnitude":
        x = np.abs(b)
    elif component == "real":
        x = np.real(b)
    elif component == "imag":
        x = np.imag(b)
    elif component == "phase":
        x = np.unwrap(np.angle(b))
    else:
        raise ValueError("component must be one of: magnitude, real, imag, phase")

    std = np.std(x)
    if std == 0:
        return np.asarray([], dtype=int)

    z = np.abs((x - np.mean(x)) / std)
    return np.where(z >= threshold)[0]


def detect_transitions(b, derivative_threshold: float = 2.0, *, component: str = "magnitude"):
    """
    Detect sharp transitions based on derivative spikes.
    """
    b = np.asarray(b)

    if b.size < 2:
        return np.asarray([], dtype=int)

    if component == "magnitude":
        x = np.abs(b)
    elif component == "real":
        x = np.real(b)
    elif component == "imag":
        x = np.imag(b)
    elif component == "phase":
        x = np.unwrap(np.angle(b))
    else:
        raise ValueError("component must be one of: magnitude, real, imag, phase")

    dx = np.diff(x)
    std = np.std(dx)

    if std == 0:
        return np.asarray([], dtype=int)

    spikes = np.abs(dx / std)
    return np.where(spikes >= derivative_threshold)[0]


def compute_energy(b):
    """
    Compute energy-like quantity sum(|b|^2).
    """
    b = np.asarray(b)
    return float(np.sum(np.abs(b) ** 2))


def compute_derivative(b, *, component: str = "magnitude"):
    """
    Compute discrete derivative of a chosen component.
    """
    b = np.asarray(b)

    if b.size < 2:
        return np.zeros_like(np.abs(b), dtype=float)

    if component == "magnitude":
        x = np.abs(b)
    elif component == "real":
        x = np.real(b)
    elif component == "imag":
        x = np.imag(b)
    elif component == "phase":
        x = np.unwrap(np.angle(b))
    else:
        raise ValueError("component must be one of: magnitude, real, imag, phase")

    db = np.zeros_like(x, dtype=float)
    db[1:] = x[1:] - x[:-1]
    db[0] = db[1]
    return db


def rolling_variance(b, window: int = 10, *, component: str = "magnitude"):
    """
    Local variance over a sliding window.
    """
    b = np.asarray(b)

    if window < 2:
        raise ValueError("window must be >= 2")

    if b.size < window:
        return np.zeros_like(np.abs(b), dtype=float)

    if component == "magnitude":
        x = np.abs(b)
    elif component == "real":
        x = np.real(b)
    elif component == "imag":
        x = np.imag(b)
    elif component == "phase":
        x = np.unwrap(np.angle(b))
    else:
        raise ValueError("component must be one of: magnitude, real, imag, phase")

    out = np.zeros_like(x, dtype=float)

    for i in range(window, len(x)):
        out[i] = np.var(x[i - window : i])

    return out


def detect_regime_changes(b, window: int = 20, *, component: str = "magnitude"):
    """
    Detect changes in variance regime.
    """
    var = rolling_variance(b, window=window, component=component)
    dv = np.diff(var)

    if dv.size == 0:
        return np.asarray([], dtype=int)

    threshold = np.std(dv) * 2.0
    return np.where(np.abs(dv) > threshold)[0]


def summarize_diagnostics(
    b,
    *,
    threshold: float = 3.0,
    theta=None,
) -> Dict[str, object]:
    """
    Full scientific diagnostic summary for a complex agencity trajectory.
    """
    b = np.asarray(b)

    if b.size == 0:
        return {}

    events = detect_events(b, threshold=threshold, component="magnitude")
    transitions = detect_transitions(b, component="magnitude")
    regime_changes = detect_regime_changes(b, component="magnitude")

    energy = compute_energy(b)
    db = compute_derivative(b, component="magnitude")

    mag = np.abs(b)
    real = np.real(b)
    imag = np.imag(b)
    phase = np.unwrap(np.angle(b))

    if theta is None:
        theta = phase
    else:
        theta = np.asarray(theta, dtype=float)

    out = {
        "n_samples": int(b.size),
        "mean_real": float(np.mean(real)),
        "mean_imag": float(np.mean(imag)),
        "mean_magnitude": float(np.mean(mag)),
        "std_magnitude": float(np.std(mag)),
        "min_magnitude": float(np.min(mag)),
        "max_magnitude": float(np.max(mag)),
        "mean_phase": float(np.mean(theta)) if theta.size else 0.0,
        "std_phase": float(np.std(theta)) if theta.size else 0.0,
        "energy": energy,
        "mean_derivative": float(np.mean(np.abs(db))),
        "event_count": int(events.size),
        "event_indices": events.tolist(),
        "transition_count": int(transitions.size),
        "transition_indices": transitions.tolist(),
        "regime": classify_regime(b, theta=theta),
        "regime_changes": regime_changes.tolist(),
    }

    return out