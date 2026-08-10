"""Scientific coherence diagnostics for AgencityLab.

This module belongs to the analysis layer. It consumes canonical outputs and
never changes the equations that produced them.

The accepted theory defines the local angular variance as

    Sigma_Theta(t) = Var(Theta(s); s in [t - tau, t]).

For a wrapped angular coordinate, each complete structurally valid local window
is unwrapped before ordinary variance is evaluated. Circular resultant-length
statistics are exposed separately as diagnostics; they are not silently
substituted for the theoretical Sigma_Theta definition.
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


def _complex_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=complex)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _valid_mask(mask, n: int) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.ndim != 1 or out.size != n:
        raise ValueError("valid_mask must be one-dimensional and match the data length")
    return out


def _validate_axis(xi, n: int) -> np.ndarray:
    axis = _real_1d(xi, name="xi")
    if axis.size != n:
        raise ValueError("xi must match the data length")
    if axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("xi must be strictly increasing")
    return axis


def phase_coherence(values, *, valid_mask=None, values_are_angles: bool = False) -> float:
    """Return the mean resultant length of an angular series.

    This is a circular-statistics diagnostic in [0, 1]. When
    ``values_are_angles=False`` the phase of the complex input is used. For
    structural coherence, callers should pass canonical ``Theta`` with
    ``values_are_angles=True`` rather than ``arg(beta)`` because the sign of J
    may rotate beta by pi without changing the structural direction.
    """
    if values_are_angles:
        theta = _real_1d(values, name="theta")
    else:
        theta = np.angle(_complex_1d(values, name="values"))
    mask = _valid_mask(valid_mask, theta.size)
    if not np.any(mask):
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * theta[mask]))))


def amplitude_coherence(b) -> float:
    """Return a dimensionless amplitude-stability diagnostic.

    The score is ``1 / (1 + coefficient_of_variation)``. It is diagnostic,
    not a canonical agencity quantity.
    """
    mag = np.abs(_complex_1d(b, name="b"))
    if mag.size == 0:
        return float("nan")
    mean = float(np.mean(mag))
    std = float(np.std(mag))
    if mean == 0.0:
        return 1.0 if std == 0.0 else 0.0
    return float(1.0 / (1.0 + std / mean))


def temporal_coherence(b) -> float:
    """Return lag-one correlation of ``|b|`` as a descriptive diagnostic."""
    mag = np.abs(_complex_1d(b, name="b"))
    if mag.size < 2:
        return float("nan")
    x = mag[:-1]
    y = mag[1:]
    if np.var(x) == 0.0 or np.var(y) == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def sigma_theta(theta, xi, tau, *, valid_mask=None) -> np.ndarray:
    """Compute the theoretical local angular variance ``Sigma_Theta``.

    For each time ``t`` with a complete interval ``[t - tau, t]``, the wrapped
    orientation samples in that interval are locally unwrapped and their
    population variance is returned. If any orientation in the interval is
    structurally undefined, or if the interval has fewer than two samples, the
    value is ``NaN``. This avoids silently computing the variance on a shortened
    subset of the theoretical time window.

    No threshold is used to decide whether a variance is "low" or "high".
    """
    theta = _real_1d(theta, name="theta")
    axis = _validate_axis(xi, theta.size)
    try:
        tau_value = float(tau)
    except Exception as exc:
        raise ValueError("tau must be a positive finite scalar") from exc
    if not np.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau must be a positive finite scalar")

    valid = _valid_mask(valid_mask, theta.size)
    out = np.full(theta.size, np.nan, dtype=float)
    start_time = axis[0]

    for i, t in enumerate(axis):
        if t - start_time < tau_value:
            continue
        left = int(np.searchsorted(axis, t - tau_value, side="left"))
        indices = np.arange(left, i + 1)
        if indices.size < 2 or not np.all(valid[indices]):
            continue
        local_theta = np.unwrap(theta[indices])
        out[i] = float(np.var(local_theta))
    return out


def orientation_statistics(M, O) -> dict[str, float]:
    """Return circular statistics of the canonical structural orientation.

    Points with ``S = hypot(M, O) = 0`` are excluded because structural
    orientation is physically undefined there even though the result container
    uses a conventional numerical angle.
    """
    M = _real_1d(M, name="M")
    O = _real_1d(O, name="O")
    if M.size != O.size:
        raise ValueError("M and O must have the same length")
    S = np.hypot(M, O)
    mask = S > 0.0
    if not np.any(mask):
        return {
            "theta_mean": float("nan"),
            "circular_variance": float("nan"),
            "phase_coherence": float("nan"),
            "valid_fraction": 0.0,
        }
    theta = np.arctan2(O[mask], M[mask])
    resultant = np.mean(np.exp(1j * theta))
    return {
        "theta_mean": float(np.angle(resultant)),
        "circular_variance": float(1.0 - np.abs(resultant)),
        "phase_coherence": float(np.abs(resultant)),
        "valid_fraction": float(np.mean(mask)),
    }


def angular_stability(M, O, *, xi=None, tau=None, window=None) -> dict[str, object]:
    """Return structural-orientation stability diagnostics.

    ``xi`` and ``tau`` select the theoretical ``Sigma_Theta`` definition.
    ``window`` is retained only as a compatibility path for the historical
    sample-count circular variance and is explicitly labelled diagnostic.
    """
    M = _real_1d(M, name="M")
    O = _real_1d(O, name="O")
    if M.size != O.size:
        raise ValueError("M and O must have the same length")
    theta = np.arctan2(O, M)
    valid = np.hypot(M, O) > 0.0

    if xi is not None or tau is not None:
        if xi is None or tau is None:
            raise ValueError("xi and tau must be provided together")
        sigma = sigma_theta(theta, xi, tau, valid_mask=valid)
        finite = np.isfinite(sigma)
        return {
            "sigma_theta": sigma,
            "sigma_theta_mean": float(np.mean(sigma[finite])) if np.any(finite) else float("nan"),
            "definition": "Var(Theta(s); s in [t-tau,t])",
            "status": "theory-derived",
        }

    if window is None:
        angles = theta[valid]
        circular = phase_coherence(angles, values_are_angles=True) if angles.size else float("nan")
        variance = 1.0 - circular if np.isfinite(circular) else float("nan")
        return {
            "sigma_theta": np.asarray([variance]),
            "sigma_theta_mean": float(variance),
            "definition": "1 - resultant_length",
            "status": "circular diagnostic compatibility",
        }

    if not isinstance(window, (int, np.integer)) or int(window) < 2:
        raise ValueError("window must be an integer >= 2")
    window = int(window)
    out = np.full(theta.size, np.nan, dtype=float)
    for i in range(window - 1, theta.size):
        segment_mask = valid[i - window + 1 : i + 1]
        segment = theta[i - window + 1 : i + 1][segment_mask]
        if segment.size < 2:
            continue
        out[i] = 1.0 - phase_coherence(segment, values_are_angles=True)
    finite = np.isfinite(out)
    return {
        "sigma_theta": out,
        "sigma_theta_mean": float(np.mean(out[finite])) if np.any(finite) else float("nan"),
        "definition": "1 - resultant_length over sample window",
        "status": "circular diagnostic compatibility",
    }


def detect_structural_plateaus(
    S,
    xi,
    *,
    slope_threshold: float,
    min_duration: float,
) -> list[dict[str, float | int]]:
    """Detect approximately flat structural-intensity intervals.

    ``slope_threshold`` and ``min_duration`` are explicit diagnostic choices;
    the theory does not provide universal numerical values for them.
    """
    S = _real_1d(S, name="S")
    axis = _validate_axis(xi, S.size)
    slope_threshold = float(slope_threshold)
    min_duration = float(min_duration)
    if not np.isfinite(slope_threshold) or slope_threshold < 0.0:
        raise ValueError("slope_threshold must be finite and non-negative")
    if not np.isfinite(min_duration) or min_duration < 0.0:
        raise ValueError("min_duration must be finite and non-negative")

    derivative = np.gradient(S, axis, edge_order=2 if S.size > 2 else 1)
    flat = np.abs(derivative) <= slope_threshold
    plateaus: list[dict[str, float | int]] = []
    start = None
    for i, is_flat in enumerate(flat):
        if is_flat and start is None:
            start = i
        at_end = i == flat.size - 1
        if start is not None and ((not is_flat) or at_end):
            end = i if is_flat and at_end else i - 1
            duration = float(axis[end] - axis[start])
            if duration >= min_duration:
                plateaus.append(
                    {
                        "start_index": int(start),
                        "end_index": int(end),
                        "start_time": float(axis[start]),
                        "end_time": float(axis[end]),
                        "duration": duration,
                        "mean_S": float(np.mean(S[start : end + 1])),
                    }
                )
            start = None
    return plateaus


def real_agencity_criterion(
    S,
    theta_variance,
    b,
    *,
    s_threshold: float = 0.0,
    theta_variance_threshold: float | None = None,
    b_threshold: float | None = None,
    min_fraction: float | None = None,
) -> dict[str, object]:
    """Evaluate the theory's real-agencity criterion as a diagnostic layer.

    The theory specifies ``S > 0``, low ``Sigma_Theta``, and significant ``|b|``
    but does not prescribe universal numerical meanings for "low" or
    "significant". Therefore this function never invents them. If either
    contextual threshold is omitted, the returned status is ``undetermined``.

    Even when local thresholds are supplied, a global Boolean is not invented:
    callers may optionally provide ``min_fraction`` to define how much of the
    evaluated interval must satisfy the local criterion. This prevents a single
    intermittent sample in noise or chaos from being silently promoted to a
    whole-record real-agencity classification.
    """
    S = _real_1d(S, name="S")
    b = _complex_1d(b, name="b")
    if S.size != b.size:
        raise ValueError("S and b must have the same length")

    sigma = np.asarray(theta_variance, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(S.size, float(sigma), dtype=float)
    if sigma.ndim != 1 or sigma.size != S.size:
        raise ValueError("theta_variance must be a scalar or match S")

    s_threshold = float(s_threshold)
    if not np.isfinite(s_threshold) or s_threshold < 0.0:
        raise ValueError("s_threshold must be finite and non-negative")

    has_structure = S > s_threshold
    finite_sigma = np.isfinite(sigma)

    stable_orientation = None
    if theta_variance_threshold is not None:
        theta_variance_threshold = float(theta_variance_threshold)
        if not np.isfinite(theta_variance_threshold) or theta_variance_threshold < 0.0:
            raise ValueError("theta_variance_threshold must be finite and non-negative")
        stable_orientation = finite_sigma & (sigma < theta_variance_threshold)

    significant_flow = None
    if b_threshold is not None:
        b_threshold = float(b_threshold)
        if not np.isfinite(b_threshold) or b_threshold < 0.0:
            raise ValueError("b_threshold must be finite and non-negative")
        significant_flow = np.abs(b) > b_threshold

    local_configured = stable_orientation is not None and significant_flow is not None
    local = None
    fraction = float("nan")
    global_value = None
    if local_configured:
        local = has_structure & stable_orientation & significant_flow
        evaluated = finite_sigma
        fraction = float(np.mean(local[evaluated])) if np.any(evaluated) else float("nan")
        status = "local criterion evaluated"
    else:
        status = "undetermined"

    if min_fraction is not None:
        min_fraction = float(min_fraction)
        if not np.isfinite(min_fraction) or not 0.0 <= min_fraction <= 1.0:
            raise ValueError("min_fraction must lie in [0, 1]")
        if not local_configured or not np.isfinite(fraction):
            global_value = None
        else:
            global_value = bool(fraction >= min_fraction)
            status = "local and global criterion evaluated"

    return {
        "status": status,
        "real_agencity": global_value,
        "local_real_agencity": local,
        "real_agencity_fraction": fraction,
        "has_structure": has_structure,
        "stable_orientation": stable_orientation,
        "significant_flow": significant_flow,
        "mean_S": float(np.mean(S)),
        "mean_abs_b": float(np.mean(np.abs(b))),
        "sigma_theta_mean": float(np.mean(sigma[finite_sigma])) if np.any(finite_sigma) else float("nan"),
        "thresholds": {
            "S_min": s_threshold,
            "Sigma_Theta_max": theta_variance_threshold,
            "abs_b_min": b_threshold,
            "min_fraction": min_fraction,
        },
        "interpretation": "diagnostic; thresholds are contextual, not canonical constants",
    }


def full_coherence(b) -> dict[str, float]:
    """Return compatibility coherence diagnostics based on the complex flux."""
    return {
        "phase_coherence": phase_coherence(b),
        "amplitude_coherence": amplitude_coherence(b),
        "temporal_coherence": temporal_coherence(b),
    }


def full_structural_coherence(M, O, b, *, xi=None, tau=None, window=None) -> dict[str, object]:
    """Return structural and flux coherence diagnostics without changing core data."""
    return {
        "orientation": orientation_statistics(M, O),
        "angular": angular_stability(M, O, xi=xi, tau=tau, window=window),
        "coherence": full_coherence(b),
    }
