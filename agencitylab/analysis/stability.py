"""
Stability helpers for AgencityLab.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .metrics import agencity_mean, agencity_variance


def stability_summary(b, epsilon: float = 1e-12) -> Dict[str, Any]:
    """
    Return a simple stability summary for the observable b.
    """
    b = np.asarray(b, dtype=float)
    var = agencity_variance(b)
    mean = agencity_mean(b)

    if b.size < 2:
        trend = 0.0
    else:
        x = np.arange(b.size, dtype=float)
        coeffs = np.polyfit(x, b, deg=1)
        trend = float(coeffs[0])

    return {
        "mean": mean,
        "variance": var,
        "trend": trend,
        "bounded_hint": bool(np.max(np.abs(b)) < 1e6) if b.size else True,
        "stable_hint": bool(abs(trend) < epsilon and var < 1e6),
    }
