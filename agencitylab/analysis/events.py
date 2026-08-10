"""Event diagnostics for AgencityLab.

Canonical quantities are never recomputed here.  The module detects features in
already-computed outputs such as peaks of dynamic intensity D.  Thresholds used
for prominence or outlier selection are diagnostic choices, not physical laws.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _real_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _select_component(b, component: str = "magnitude") -> np.ndarray:
    b = np.asarray(b, dtype=complex)
    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")
    if component == "magnitude":
        return np.abs(b)
    if component == "real":
        return np.real(b)
    if component == "imag":
        return np.imag(b)
    if component == "phase":
        return np.unwrap(np.angle(b))
    raise ValueError("component must be one of: magnitude, real, imag, phase")


def detect_dynamic_peaks(D, *, prominence: float | None = None, distance: int | None = None) -> np.ndarray:
    """Return local maxima of canonical dynamic intensity ``D``.

    With ``prominence=None`` every strict local maximum accepted by
    :func:`scipy.signal.find_peaks` is returned.  Supplying a prominence or
    sample distance is an explicit diagnostic filter.
    """
    D = _real_1d(D, name="D")
    kwargs = {}
    if prominence is not None:
        prominence = float(prominence)
        if not np.isfinite(prominence) or prominence < 0.0:
            raise ValueError("prominence must be finite and non-negative")
        kwargs["prominence"] = prominence
    if distance is not None:
        if not isinstance(distance, (int, np.integer)) or int(distance) < 1:
            raise ValueError("distance must be an integer >= 1")
        kwargs["distance"] = int(distance)
    indices, _ = find_peaks(D, **kwargs)
    return np.asarray(indices, dtype=int)


def dynamic_peak_summary(D, xi=None, *, prominence: float | None = None, distance: int | None = None) -> dict[str, object]:
    """Summarize peaks of D without assigning physical significance to them."""
    D = _real_1d(D, name="D")
    indices = detect_dynamic_peaks(D, prominence=prominence, distance=distance)
    if xi is None:
        times = indices.astype(float)
    else:
        axis = _real_1d(xi, name="xi")
        if axis.size != D.size:
            raise ValueError("xi must match D")
        times = axis[indices]
    return {
        "count": int(indices.size),
        "indices": indices,
        "times": np.asarray(times, dtype=float),
        "values": D[indices],
        "prominence_threshold": prominence,
        "distance_samples": distance,
        "status": "diagnostic",
    }


def detect_events(
    b,
    *,
    threshold: float = 3.0,
    component: str = "magnitude",
    verbose: bool = False,
):
    """Compatibility z-score outlier detector on a selected component of b.

    The default ``3.0`` is a historical software diagnostic and must not be
    interpreted as a canonical agencity threshold.
    """
    x = _select_component(b, component=component)
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
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
    """Return the historical b-outlier summary, explicitly labelled diagnostic."""
    x = _select_component(b, component=component)
    idx = detect_events(b, threshold=threshold, component=component, verbose=verbose)
    values = x[idx] if idx.size else np.asarray([], dtype=float)
    return {
        "component": component,
        "threshold": float(threshold),
        "event_count": int(idx.size),
        "event_indices": idx.tolist(),
        "event_values": values.tolist(),
        "event_rate": float(idx.size / max(1, x.size)),
        "status": "legacy diagnostic compatibility",
    }
