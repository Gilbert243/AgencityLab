"""
Transition detection for AgencityLab.

Detects sharp local changes from derivatives and variance shifts.
"""

from __future__ import annotations

from typing import Dict

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


def detect_transitions(
    b,
    *,
    derivative_threshold: float = 2.0,
    window: int = 20,
    component: str = "magnitude",
    verbose: bool = False,
):
    """
    Detect changes in regime by combining derivative spikes and local variance shifts.
    """
    x = _select_component(b, component=component)

    if x.size < 2:
        return np.asarray([], dtype=int)

    dx = np.diff(x)
    dstd = float(np.std(dx))

    if dstd == 0.0:
        return np.asarray([], dtype=int)

    spikes = np.where(np.abs(dx / dstd) >= derivative_threshold)[0]

    if x.size < 2 * window:
        idx = spikes
    else:
        rolling = np.zeros_like(x, dtype=float)
        for i in range(window, x.size):
            rolling[i] = np.var(x[i - window : i])
        dvar = np.diff(rolling)
        vstd = float(np.std(dvar)) if dvar.size else 0.0
        var_spikes = np.where(np.abs(dvar) >= 2.0 * vstd)[0] if vstd > 0 else np.asarray([], dtype=int)
        idx = np.unique(np.concatenate([spikes, var_spikes])) if spikes.size or var_spikes.size else np.asarray([], dtype=int)

    if verbose:
        print(f"[transitions] component={component}, count={len(idx)}")

    return idx


def transition_summary(
    b,
    *,
    derivative_threshold: float = 2.0,
    window: int = 20,
    component: str = "magnitude",
    verbose: bool = False,
) -> Dict[str, object]:
    idx = detect_transitions(
        b,
        derivative_threshold=derivative_threshold,
        window=window,
        component=component,
        verbose=verbose,
    )

    out = {
        "component": component,
        "derivative_threshold": float(derivative_threshold),
        "window": int(window),
        "transition_count": int(idx.size),
        "transition_indices": idx.tolist(),
    }

    if verbose:
        print("[transitions] ---")
        for k, v in out.items():
            print(f"[transitions] {k}: {v}")

    return out