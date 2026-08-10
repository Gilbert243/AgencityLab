"""
Event detection for AgencityLab.

Detects strong local deviations in the chosen component of b(t).
"""

from __future__ import annotations

import numpy as np


def _select_component(b, component: str = "magnitude"):
    b = np.asarray(b)
    if component == "magnitude":
        return np.abs(b)
    if component == "real":
        return np.real(b)
    if component == "imag":
        return np.imag(b)
    if component == "phase":
        return np.unwrap(np.angle(b))
    raise ValueError("component must be one of: magnitude, real, imag, phase")


def detect_events(
    b,
    *,
    threshold: float = 3.0,
    component: str = "magnitude",
    verbose: bool = False,
):
    """
    Detect strong outliers using z-score.
    Returns indices of event samples.
    """
    x = _select_component(b, component=component)

    if x.size == 0:
        return np.asarray([], dtype=int)

    std = float(np.std(x))
    if std == 0.0:
        return np.asarray([], dtype=int)

    z = np.abs((x - np.mean(x)) / std)
    idx = np.where(z >= threshold)[0]

    if verbose:
        print(f"[events] component={component}, threshold={threshold}, count={len(idx)}")

    return idx


def event_summary(
    b,
    *,
    threshold: float = 3.0,
    component: str = "magnitude",
    verbose: bool = False,
):
    """
    Return a compact event summary.
    """
    x = _select_component(b, component=component)
    idx = detect_events(b, threshold=threshold, component=component, verbose=verbose)

    values = x[idx] if idx.size else np.asarray([], dtype=float)

    out = {
        "component": component,
        "threshold": float(threshold),
        "event_count": int(idx.size),
        "event_indices": idx.tolist(),
        "event_values": values.tolist(),
        "event_rate": float(idx.size / max(1, x.size)),
    }

    if verbose:
        print("[events] ---")
        for k, v in out.items():
            print(f"[events] {k}: {v}")

    return out