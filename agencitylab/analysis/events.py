"""Event diagnostics for AgencityLab.

Canonical quantities are never recomputed here. The module detects features in
already-computed outputs such as peaks of dynamic intensity D. Thresholds used
for prominence or outlier selection are diagnostic choices, not physical laws.
"""

from __future__ import annotations

import numpy as np


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


def _unfiltered_local_maxima(values: np.ndarray) -> np.ndarray:
    """Return SciPy-compatible unfiltered one-dimensional local maxima.

    Endpoints are excluded. For a flat maximum, the middle sample is returned,
    rounded down for an even plateau. The implementation is O(N), requires only
    NumPy, and is used only when no optional diagnostic filter is requested.
    """
    if values.size < 3:
        return np.asarray([], dtype=int)

    peaks: list[int] = []
    index = 1
    last = values.size - 1
    while index < last:
        if values[index] <= values[index - 1]:
            index += 1
            continue

        plateau_end = index
        while plateau_end < last and values[plateau_end + 1] == values[index]:
            plateau_end += 1

        if plateau_end < last and values[plateau_end] > values[plateau_end + 1]:
            peaks.append((index + plateau_end) // 2)
        index = plateau_end + 1

    return np.asarray(peaks, dtype=int)


def detect_dynamic_peaks(
    D,
    *,
    prominence: float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """Return local maxima of canonical dynamic intensity ``D``.

    With no filters, local maxima are found by the NumPy-only O(N) path and flat
    peaks follow the historical SciPy midpoint convention. Explicit prominence
    or distance filters retain the established :func:`scipy.signal.find_peaks`
    semantics and therefore require the ``scientific`` optional dependency.
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

    if not kwargs:
        return _unfiltered_local_maxima(D)

    try:
        from scipy.signal import find_peaks
    except ImportError as exc:  # pragma: no cover - exercised by clean-install CI
        raise ImportError(
            "filtered dynamic peak detection requires SciPy; install "
            "AgencityLab with the scientific extra"
        ) from exc

    indices, _ = find_peaks(D, **kwargs)
    return np.asarray(indices, dtype=int)


def dynamic_peak_summary(
    D,
    xi=None,
    *,
    prominence: float | None = None,
    distance: int | None = None,
) -> dict[str, object]:
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
