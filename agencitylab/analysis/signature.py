"""
signature.py

Agencity signature (multi-scale fingerprint of a system).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Iterable

from .multi_scale import agencity_spectrum


def agencity_signature(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
) -> Dict[str, Any]:
    """
    Compute the agencity signature of a system.

    Returns
    -------
    dict with:
        - tau
        - b_mean
        - b_std
        - peak_tau
        - peak_value
        - growth_trend
        - signature_type
    """

    spec = agencity_spectrum(xi, u, taus)

    tau_vals = np.array([s["tau"] for s in spec])
    b_mean = np.array([s["b_mean"] for s in spec])
    b_std = np.array([s["b_std"] for s in spec])

    # 🔹 Peak detection
    peak_idx = int(np.argmax(b_std))
    peak_tau = tau_vals[peak_idx]
    peak_value = b_std[peak_idx]

    # 🔹 Growth trend (simple slope)
    slope = np.polyfit(tau_vals, b_std, 1)[0]

    # 🔹 Classification
    if np.allclose(b_std, 0):
        signature_type = "inactive"
    elif slope > 0 and peak_tau == tau_vals[-1]:
        signature_type = "scale-expanding"
    elif slope < 0:
        signature_type = "scale-damped"
    else:
        signature_type = "structured"

    return {
        "tau": tau_vals,
        "b_mean": b_mean,
        "b_std": b_std,
        "peak_tau": float(peak_tau),
        "peak_value": float(peak_value),
        "growth_trend": float(slope),
        "signature_type": signature_type,
    }


def print_signature(sig: Dict[str, Any]) -> None:
    """
    Pretty print signature.
    """

    print("\n=== AGENCY SIGNATURE ===")

    print(f"Type        : {sig['signature_type']}")
    print(f"Peak tau    : {sig['peak_tau']:.3f}")
    print(f"Peak b_std  : {sig['peak_value']:.4f}")
    print(f"Trend       : {sig['growth_trend']:.4f}")