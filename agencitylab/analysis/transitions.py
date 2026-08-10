"""Transition diagnostics for canonical Agencity outputs."""

from __future__ import annotations

from typing import Dict

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


def detect_agencity_zeros(S, J, *, atol: float = 0.0) -> np.ndarray:
    """Return samples satisfying the theory's zero condition for beta/b.

    For strictly positive characteristic power,

        b = 0 iff beta = 0 iff S = 0 or (S > 0 and J = 0).

    The default is exact.  A positive ``atol`` is an explicitly requested
    numerical diagnostic tolerance and does not redefine physical zero.
    """
    S = _real_1d(S, name="S")
    J = _real_1d(J, name="J")
    if S.size != J.size:
        raise ValueError("S and J must have the same length")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    if atol == 0.0:
        mask = (S == 0.0) | ((S > 0.0) & (J == 0.0))
    else:
        mask = (S <= atol) | ((S > atol) & (np.abs(J) <= atol))
    return np.flatnonzero(mask)


def critical_surface_crossings(D, S) -> np.ndarray:
    """Return indices bracketing crossings of the critical surface ``D = S``.

    A returned index ``i`` means the crossing occurs at sample ``i`` exactly or
    in the interval ``[i, i+1]``.  Sign changes are used rather than an arbitrary
    near-zero threshold.
    """
    D = _real_1d(D, name="D")
    S = _real_1d(S, name="S")
    if D.size != S.size:
        raise ValueError("D and S must have the same length")
    if D.size == 0:
        return np.asarray([], dtype=int)
    delta = D - S
    exact = np.flatnonzero(delta == 0.0)
    if D.size < 2:
        return exact.astype(int)
    crossing = np.flatnonzero(delta[:-1] * delta[1:] < 0.0)
    return np.unique(np.concatenate([exact, crossing])).astype(int)


def detect_theta_jumps(theta, *, threshold: float, valid_mask=None) -> np.ndarray:
    """Detect structural-orientation jumps larger than an explicit angle.

    ``threshold`` is contextual and diagnostic.  Minimal wrapped angular
    differences are used so the ±pi branch cut is not mistaken for a physical
    jump.
    """
    theta = _real_1d(theta, name="theta")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > np.pi:
        raise ValueError("threshold must be finite and lie in [0, pi]")
    if theta.size < 2:
        return np.asarray([], dtype=int)
    if valid_mask is None:
        valid = np.ones(theta.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim != 1 or valid.size != theta.size:
            raise ValueError("valid_mask must match theta")
    delta = np.angle(np.exp(1j * (theta[1:] - theta[:-1])))
    pair_valid = valid[:-1] & valid[1:]
    return np.flatnonzero(pair_valid & (np.abs(delta) >= threshold)).astype(int)


def zero_summary(S, J, xi=None, *, atol: float = 0.0) -> dict[str, object]:
    """Summarize canonical agencity zeros and the tolerance policy."""
    indices = detect_agencity_zeros(S, J, atol=atol)
    if xi is None:
        times = indices.astype(float)
    else:
        axis = _real_1d(xi, name="xi")
        if axis.size != np.asarray(S).size:
            raise ValueError("xi must match S and J")
        times = axis[indices]
    return {
        "count": int(indices.size),
        "indices": indices,
        "times": np.asarray(times, dtype=float),
        "atol": float(atol),
        "definition": "S=0 or (S>0 and J=0)",
        "status": "exact theory condition" if atol == 0.0 else "numerical diagnostic tolerance",
    }


def detect_transitions(
    b,
    *,
    derivative_threshold: float = 2.0,
    window: int = 20,
    component: str = "magnitude",
    verbose: bool = False,
):
    """Historical derivative/variance transition heuristic for compatibility."""
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
        var_spikes = (
            np.where(np.abs(dvar) >= 2.0 * vstd)[0]
            if vstd > 0.0
            else np.asarray([], dtype=int)
        )
        idx = (
            np.unique(np.concatenate([spikes, var_spikes]))
            if spikes.size or var_spikes.size
            else np.asarray([], dtype=int)
        )
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
    """Return the historical b-transition summary, explicitly labelled heuristic."""
    idx = detect_transitions(
        b,
        derivative_threshold=derivative_threshold,
        window=window,
        component=component,
        verbose=verbose,
    )
    return {
        "component": component,
        "derivative_threshold": float(derivative_threshold),
        "window": int(window),
        "transition_count": int(idx.size),
        "transition_indices": idx.tolist(),
        "status": "legacy heuristic compatibility",
    }
