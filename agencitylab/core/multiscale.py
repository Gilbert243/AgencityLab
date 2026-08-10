"""Multiscale, window, discrete, and multivariate Theory of Agencity tools.

Volume 2 keeps the CRM width ``w > 0`` distinct from the characteristic time
``tau`` and often uses the convenient convention ``w = tau``. This module
implements explicit ``b(t, tau)`` studies, Chapter 13 window selection, and the
Pc-weighted multivariate construction without introducing a second theory.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .activation import activation, reduced_coordinate
from .activity import activity
from .agencity import agencity
from .beta import compute_beta
from .intensity import compute_intensities
from .memory import memory
from .normalization import normalize_signal
from .organization import organization
from .validation import (
    is_exactly_constant,
    validate_axis,
    validate_positive_scalar,
    validate_signal,
)


def _sample_step(axis: np.ndarray) -> float:
    axis = validate_axis(axis)
    diffs = np.diff(axis)
    step = float(diffs[0])
    tolerance = np.finfo(float).eps * max(1.0, abs(step)) * 64.0
    if not np.allclose(diffs, step, rtol=1e-10, atol=tolerance):
        raise ValueError("discrete multiscale extensions require uniformly sampled coordinates")
    return step


def _window_samples(window: float, axis: np.ndarray) -> int:
    window = validate_positive_scalar(window, name="w")
    step = _sample_step(axis)
    samples = int(round(window / step))
    if samples < 1:
        raise ValueError("w is smaller than one sampling interval")
    represented = samples * step
    tolerance = max(np.finfo(float).eps * max(1.0, abs(window)) * 128.0, step * 1e-9)
    if not np.isclose(represented, window, rtol=1e-9, atol=tolerance):
        raise ValueError("w must be an integer multiple of the sampling interval")
    return samples


def _power_profile(P_c, xi: np.ndarray):
    candidate = P_c(xi) if callable(P_c) else P_c
    try:
        arr = np.asarray(candidate, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("P_c must be numeric") from exc
    if arr.ndim == 0:
        value = float(arr)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("P_c must be strictly positive and finite")
        return value
    if arr.ndim != 1 or arr.shape != xi.shape:
        raise ValueError("sampled P_c must be one-dimensional and match xi")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("sampled P_c must contain strictly positive finite values")
    return arr


def compute_scale_response(
    xi,
    u,
    *,
    tau: float,
    w: float,
    A_ref: float,
    P_c,
):
    """Compute one scalar response at explicit theoretical ``tau`` and ``w``.

    Normalization, reduced derivatives, intensities, contrast, orientation,
    ``beta`` and ``b`` follow the same scalar equations. ``tau`` and the CRM
    width ``w`` remain separate parameters as specified in Volume 2.
    """
    xi = validate_axis(xi)
    u = validate_signal(u, name="u").ravel()
    if u.shape != xi.shape:
        raise ValueError("xi and u must have the same shape")
    tau = validate_positive_scalar(tau, name="tau")
    w = validate_positive_scalar(w, name="w")
    _window_samples(w, xi)
    P_eff = _power_profile(P_c, xi)
    u_star, A_ref_used = normalize_signal(u, A_ref=A_ref, method="canonical")
    t_star = reduced_coordinate(xi, tau)

    if is_exactly_constant(u):
        zeros = np.zeros_like(u_star, dtype=float)
        complex_zeros = np.zeros_like(u_star, dtype=complex)
        X_star = zeros.copy()
        A_star = zeros.copy()
        M = zeros.copy()
        O = zeros.copy()
        D = zeros.copy()
        S = zeros.copy()
        J = zeros.copy()
        U = complex_zeros.copy()
        beta = complex_zeros.copy()
        b = complex_zeros.copy()
    else:
        X_star = activation(u_star, axis=t_star)
        A_star = activity(X_star, axis=t_star)
        M = memory(u_star, tau, axis=xi, window=w)
        O = organization(u_star, X_star, tau, axis=xi, window=w)
        D, S = compute_intensities(X_star, A_star, M, O)
        J, U, beta = compute_beta(D, S, M, O)
        b = agencity(beta, P_c=P_eff)

    return {
        "xi": xi,
        "u": u,
        "u_star": u_star,
        "t_star": t_star,
        "tau": float(tau),
        "w": float(w),
        "A_ref": float(A_ref_used),
        "P_c": P_eff,
        "X_star": X_star,
        "A_star": A_star,
        "M": M,
        "O": O,
        "D": D,
        "S": S,
        "J": J,
        "U": U,
        "theta": np.angle(U),
        "beta": beta,
        "b": b,
        "window_mode": "explicit w" if w != tau else "w=tau convention",
    }


def _resolve_windows(taus: np.ndarray, windows) -> np.ndarray:
    if windows is None:
        return taus.copy()
    arr = np.asarray(windows, dtype=float)
    if arr.ndim == 0:
        arr = np.full(taus.size, float(arr), dtype=float)
    if arr.ndim != 1 or arr.size != taus.size:
        raise ValueError("windows must be a scalar or have one value per tau")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("all windows must be strictly positive and finite")
    return arr


def agencity_spectrum(
    xi,
    u,
    taus: Iterable[float],
    *,
    A_ref: float,
    P_c,
    windows=None,
    return_full: bool = False,
):
    """Compute the time-resolved multiscale spectrum ``b(t, tau)``.

    By default each scale uses the common convention ``w = tau``. Passing
    ``windows`` keeps the two theoretical parameters explicit and independent.
    """
    taus = np.asarray(list(taus), dtype=float)
    if taus.ndim != 1 or taus.size == 0:
        raise ValueError("taus must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(taus)) or np.any(taus <= 0.0):
        raise ValueError("all taus must be strictly positive and finite")
    windows_arr = _resolve_windows(taus, windows)

    responses = [
        compute_scale_response(xi, u, tau=float(tau), w=float(w), A_ref=A_ref, P_c=P_c)
        for tau, w in zip(taus, windows_arr)
    ]
    b = np.stack([item["b"] for item in responses], axis=0)
    beta = np.stack([item["beta"] for item in responses], axis=0)
    J = np.stack([item["J"] for item in responses], axis=0)
    S = np.stack([item["S"] for item in responses], axis=0)

    out = {
        "tau": taus,
        "w": windows_arr,
        "b": b,
        "beta": beta,
        "b_mean": np.mean(np.abs(b), axis=1),
        "b_rms": np.sqrt(np.mean(np.abs(b) ** 2, axis=1)),
        "beta_mean": np.mean(np.abs(beta), axis=1),
        "J_mean": np.mean(J, axis=1),
        "S_mean": np.mean(S, axis=1),
        "window_mode": "w=tau" if np.array_equal(taus, windows_arr) else "explicit independent w",
        "scientific_boundary": (
            "tau, w, sampling interval, and multiscale scanning are distinct theoretical/numerical objects"
        ),
    }
    if return_full:
        out["responses"] = responses
    return out


def _phi2(theta: np.ndarray, S: np.ndarray, samples: int) -> float:
    """Discrete Chapter 13 angular-stability functional on defined orientations."""
    values = []
    valid = S > 0.0
    for end in range(samples - 1, theta.size):
        start = end - samples + 1
        if not np.all(valid[start : end + 1]):
            continue
        segment = np.unwrap(theta[start : end + 1])
        values.append(float(np.var(segment)))
    return float(np.mean(values)) if values else float("inf")


def _candidate_windows(xi: np.ndarray, candidates, n_candidates: int) -> np.ndarray:
    step = _sample_step(xi)
    max_samples = xi.size // 2
    if max_samples < 1:
        raise ValueError("signal is too short for window optimisation")
    if candidates is None:
        raw = np.geomspace(step, max_samples * step, max(2, int(n_candidates)))
    else:
        raw = np.asarray(list(candidates), dtype=float)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError("candidates must be a non-empty one-dimensional sequence")
    sample_counts = np.unique(np.clip(np.rint(raw / step).astype(int), 1, max_samples))
    return sample_counts.astype(float) * step


def optimize_memory_window(
    xi,
    u,
    *,
    tau: float,
    A_ref: float,
    P_c,
    candidates=None,
    n_candidates: int = 24,
):
    """Select ``w`` by the theory's Chapter 13 angular-stability criterion ``Phi2``.

    Candidate widths are represented by integer sample counts as required by the
    discrete CRM. A candidate with no complete interval on which ``S > 0`` has
    undefined structural orientation and receives an infinite operational score;
    this avoids manufacturing angular coherence from the ``S = 0`` convention.
    """
    xi = validate_axis(xi)
    u = validate_signal(u, name="u").ravel()
    windows = _candidate_windows(xi, candidates, n_candidates)
    scores = []
    phi1 = []
    eligible = []

    for w in windows:
        response = compute_scale_response(xi, u, tau=tau, w=float(w), A_ref=A_ref, P_c=P_c)
        n = _window_samples(float(w), xi)
        score = _phi2(response["theta"], response["S"], n)
        start = 2 * n - 1
        contrast = (
            float(np.mean(np.abs(response["J"][start:])))
            if start < response["J"].size
            else float("nan")
        )
        scores.append(score)
        phi1.append(contrast)
        eligible.append(bool(np.isfinite(score)))

    scores_arr = np.asarray(scores, dtype=float)
    eligible_arr = np.asarray(eligible, dtype=bool)
    if not np.any(eligible_arr):
        raise ValueError("no candidate window has a complete interval with defined structural orientation")
    best_index = int(np.argmin(scores_arr))

    return {
        "w_opt": float(windows[best_index]),
        "criterion": "Phi2 angular stability",
        "candidate_w": windows,
        "phi2": scores_arr,
        "phi1_mean_abs_contrast": np.asarray(phi1, dtype=float),
        "eligible": eligible_arr,
        "best_index": best_index,
        "tau": float(tau),
        "selection_status": "theory-defined Chapter 13 window optimisation",
        "numerical_note": (
            "candidates without defined structural orientation are excluded rather than treated as zero variance"
        ),
    }


def _component_parameter(value, n_components: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_components, float(arr), dtype=float)
    if arr.ndim != 1 or arr.size != n_components:
        raise ValueError(f"{name} must be scalar or have one value per component")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain strictly positive finite values")
    return arr


def _component_power(P_c, xi: np.ndarray, n_components: int) -> np.ndarray:
    candidate = P_c(xi) if callable(P_c) else P_c
    arr = np.asarray(candidate, dtype=float)
    if arr.ndim == 0:
        arr = np.full((n_components, xi.size), float(arr), dtype=float)
    elif arr.ndim == 1:
        if arr.size != n_components:
            raise ValueError("one-dimensional multivariate P_c must have one value per component")
        arr = np.repeat(arr[:, None], xi.size, axis=1)
    elif arr.ndim == 2:
        if arr.shape == (xi.size, n_components):
            arr = arr.T
        elif arr.shape != (n_components, xi.size):
            raise ValueError("sampled multivariate P_c must have shape (components, samples) or transpose")
    else:
        raise ValueError("multivariate P_c must be scalar, component vector, or sampled matrix")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("multivariate P_c must contain strictly positive finite values")
    return arr


def multivariate_agencity(
    xi,
    u,
    *,
    A_ref,
    tau,
    P_c,
    w=None,
    sample_axis: int = 0,
):
    """Compute the theory's Pc-weighted multivariate construction.

    Each observable component is processed by the scalar equations. The total
    flux is ``sum_k P_c,k beta_k`` and the aggregate state is the pointwise
    Pc-weighted mean ``beta_multi``.
    """
    xi = validate_axis(xi)
    data = np.asarray(u, dtype=float)
    if data.ndim != 2:
        raise ValueError("multivariate u must be a two-dimensional array")
    if sample_axis not in {0, 1}:
        raise ValueError("sample_axis must be 0 or 1")
    if sample_axis == 1:
        data = data.T
    if data.shape[0] != xi.size:
        raise ValueError("the sample dimension of u must match xi")
    if not np.all(np.isfinite(data)):
        raise ValueError("multivariate u must contain only finite values")

    n_components = data.shape[1]
    A_refs = _component_parameter(A_ref, n_components, "A_ref")
    taus = _component_parameter(tau, n_components, "tau")
    windows = taus.copy() if w is None else _component_parameter(w, n_components, "w")
    powers = _component_power(P_c, xi, n_components)

    responses = []
    for k in range(n_components):
        responses.append(
            compute_scale_response(
                xi,
                data[:, k],
                tau=float(taus[k]),
                w=float(windows[k]),
                A_ref=float(A_refs[k]),
                P_c=powers[k],
            )
        )

    beta_components = np.stack([item["beta"] for item in responses], axis=0)
    b_components = powers * beta_components
    total_power = np.sum(powers, axis=0)
    b_total = np.sum(b_components, axis=0)
    beta_multi = b_total / total_power

    return {
        "xi": xi,
        "n_components": n_components,
        "A_ref": A_refs,
        "tau": taus,
        "w": windows,
        "P_c_components": powers,
        "P_c_total": total_power,
        "beta_components": beta_components,
        "b_components": b_components,
        "beta_multi": beta_multi,
        "b_total": b_total,
        "components": responses,
        "aggregation": "Pc-weighted beta; vector-additive total flux",
        "scientific_boundary": "Volume 2 multivariate construction; each component uses scalar Agencity",
    }


def multiscale_agencity(
    t,
    u,
    *,
    scales=None,
    tau=None,
    P_c=None,
    A_ref=None,
    w=None,
    return_full=False,
    **legacy,
):
    """Compatibility wrapper for the historical multiplicative-scale API.

    Unlike the pre-v0.6 implementation, this wrapper never infers ``A_ref``,
    ``tau`` or ``P_c`` from the observed signal and never compresses ``w``.
    """
    unsupported = {key: value for key, value in legacy.items() if value not in {None, False, "auto"}}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported legacy multiscale option(s): {names}")
    if tau is None or isinstance(tau, str):
        raise ValueError("multiscale_agencity requires an explicit physical tau")
    if P_c is None or isinstance(P_c, str):
        raise ValueError("multiscale_agencity requires explicit P_c")
    if A_ref is None or isinstance(A_ref, str):
        raise ValueError("multiscale_agencity requires explicit A_ref")
    if scales is None:
        scales = np.geomspace(0.5, 2.5, 12)
    scales = np.asarray(scales, dtype=float)
    taus = float(tau) * scales
    windows = None if w is None else np.full(taus.size, float(w), dtype=float)
    out = agencity_spectrum(t, u, taus, A_ref=A_ref, P_c=P_c, windows=windows, return_full=return_full)
    out["scale"] = scales
    return out
