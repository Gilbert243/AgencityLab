"""
Diagnostic helpers for AgencityLab.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def detect_events(b, threshold: float = 3.0):
    """
    Detect strong outliers in the Agencity observable.
    """
    b = np.asarray(b, dtype=float)
    if b.size == 0:
        return np.asarray([], dtype=int)

    mean = np.mean(b)
    std = np.std(b)
    if std == 0:
        return np.asarray([], dtype=int)

    z = np.abs((b - mean) / std)
    return np.where(z >= threshold)[0]


def summarize_diagnostics(b, threshold: float = 3.0) -> Dict[str, object]:
    """
    Return a compact diagnostic summary for a trajectory.
    """
    b = np.asarray(b, dtype=float)
    events = detect_events(b, threshold=threshold)

    return {
        "n_samples": int(b.size),
        "mean": float(np.mean(b)) if b.size else 0.0,
        "std": float(np.std(b)) if b.size else 0.0,
        "min": float(np.min(b)) if b.size else 0.0,
        "max": float(np.max(b)) if b.size else 0.0,
        "event_count": int(events.size),
        "event_indices": events.tolist(),
    }
