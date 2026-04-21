"""
Computation entry points for AgencityLab.

This module converts raw inputs into the canonical AgencityResult object.
It uses lazy imports so that optional scientific extras remain optional.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agencitylab.data.pipeline import prepare_signal
from agencitylab.models.metadata import ExperimentMetadata
from agencitylab.models.result import AgencityResult


def _import_first(module_name: str, candidates: tuple[str, ...]):
    """Import the first available attribute from a module."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _normalize(u: np.ndarray, method: str = "zscore", epsilon: float = 1e-12) -> np.ndarray:
    """Normalize the signal using the core layer when available."""
    fn = _import_first(
        "agencitylab.core.normalization",
        ("normalize_signal", "normalize", "normalize_u", "compute_normalization"),
    )

    if callable(fn):
        for args in (
            (u, method, epsilon),
            (u, method),
            (u,),
        ):
            try:
                out = fn(*args)

                # 🔥 FIX CRITIQUE
                if isinstance(out, tuple):
                    out = out[0]  # on garde seulement u*

                return np.asarray(out, dtype=float)

            except TypeError:
                continue

    from agencitylab.data.transforms.normalize import normalize_signal
    return np.asarray(normalize_signal(u, method=method, epsilon=epsilon), dtype=float)

def _activation(u_star: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Compute X* with the core layer when available."""
    fn = _import_first(
        "agencitylab.core.activation",
        ("compute_activation", "activation", "compute_X_star"),
    )
    if callable(fn):
        step = float(np.mean(np.diff(xi))) if xi.size > 1 else 1.0
        for args in ((u_star, step), (u_star,),):
            try:
                return np.asarray(fn(*args), dtype=float)
            except TypeError:
                continue

    if xi.size < 2:
        return np.zeros_like(u_star)
    step = float(np.mean(np.diff(xi)))
    out = np.zeros_like(u_star, dtype=float)
    out[1:-1] = (u_star[2:] - u_star[:-2]) / (2.0 * step)
    out[0] = (u_star[1] - u_star[0]) / step
    out[-1] = (u_star[-1] - u_star[-2]) / step
    return out


def _activity(X_star: np.ndarray, window: int = 1) -> np.ndarray:
    """Compute A* with the core layer when available."""
    fn = _import_first(
        "agencitylab.core.activity",
        ("compute_activity", "activity", "compute_A_star"),
    )
    if callable(fn):
        for args in ((X_star, window), (X_star,),):
            try:
                return np.asarray(fn(*args), dtype=float)
            except TypeError:
                continue

    n = X_star.size
    out = np.zeros_like(X_star, dtype=float)
    for i in range(n):
        ip = min(i + window, n - 1)
        im = max(i - window, 0)
        out[i] = X_star[ip] - X_star[im]
    return out


def _estimate_tau(xi: np.ndarray, X_star: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-12) -> float:
    """Estimate the characteristic scale tau."""
    fn = _import_first(
        "agencitylab.core.tau",
        ("estimate_tau", "compute_tau", "tau_from_autocorr"),
    )
    if callable(fn):
        for args in ((X_star,), (X_star, xi),):
            try:
                return float(fn(*args, threshold=threshold, epsilon=epsilon))
            except TypeError:
                continue

    if X_star.size < 2:
        return 1.0
    centered = X_star - np.mean(X_star)
    if np.allclose(centered, 0.0):
        return float(np.mean(np.diff(xi)) if xi.size > 1 else 1.0)
    ac = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if ac[0] <= epsilon:
        return float(np.mean(np.diff(xi)) if xi.size > 1 else 1.0)
    ac = ac / ac[0]
    idx = np.where(ac <= threshold)[0]
    if idx.size == 0:
        return float(np.mean(np.diff(xi)) if xi.size > 1 else 1.0)
    step = float(np.mean(np.diff(xi))) if xi.size > 1 else 1.0
    return max(step, float(idx[0]) * step)


def _crm(values: np.ndarray, window: int = 1, epsilon: float = 1e-12) -> np.ndarray:
    """Compute causal moving correlation."""
    fn = _import_first(
        "agencitylab.core.crm",
        ("causal_moving_correlation", "crm", "compute_crm"),
    )
    if callable(fn):
        for args in ((values, window, epsilon), (values, window), (values,),):
            try:
                return np.asarray(fn(*args), dtype=float)
            except TypeError:
                continue

    values = np.asarray(values, dtype=float)
    if values.size < 2 * window:
        return np.zeros_like(values)

    result = np.zeros_like(values, dtype=float)
    for i in range(2 * window - 1, values.size):
        a = values[i - 2 * window + 1 : i - window + 1]
        b = values[i - window + 1 : i + 1]
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
        result[i] = 0.0 if denom < epsilon else float(np.dot(a0, b0) / denom)
    return np.clip(result, -1.0, 1.0)


def _memory(A_star: np.ndarray, crm_window: int = 1) -> np.ndarray:
    """Compute M from A*."""
    fn = _import_first(
        "agencitylab.core.memory",
        ("compute_memory", "memory", "compute_M"),
    )
    if callable(fn):
        for args in ((A_star, crm_window), (A_star,),):
            try:
                return np.asarray(fn(*args), dtype=float)
            except TypeError:
                continue
    return np.tanh(_crm(A_star, window=crm_window))


def _organization(X_star: np.ndarray, crm_window: int = 1) -> np.ndarray:
    """Compute O from X*."""
    fn = _import_first(
        "agencitylab.core.organization",
        ("compute_organization", "organization", "compute_O"),
    )
    if callable(fn):
        for args in ((X_star, crm_window), (X_star,),):
            try:
                return np.asarray(fn(*args), dtype=float)
            except TypeError:
                continue
    return np.tanh(_crm(X_star, window=crm_window))


def _beta(X_star: np.ndarray, A_star: np.ndarray, M: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Compute beta from the canonical formula."""
    fn = _import_first(
        "agencitylab.core.beta",
        ("compute_beta", "beta"),
    )
    if callable(fn):
        for args in ((X_star, A_star, M, O),):
            try:
                return np.asarray(fn(*args), dtype=float)
            except TypeError:
                continue
    return np.tanh(X_star * (1.0 + A_star)) * np.tanh(M + O)


def _power(P_c: Any, xi: np.ndarray, X_star: np.ndarray, A_star: np.ndarray) -> np.ndarray:
    """Normalize characteristic power into a 1D array."""
    if callable(P_c):
        P_c = P_c(xi, X_star, A_star)
    P_c = np.asarray(P_c, dtype=float)
    if P_c.ndim == 0:
        return np.full_like(xi, float(P_c), dtype=float)
    if P_c.ndim != 1 or P_c.shape[0] != xi.shape[0]:
        raise ValueError("P_c must be a scalar or a 1D array matching xi.")
    return P_c.astype(float)


def _agencity(beta: np.ndarray, P_c: np.ndarray, tau: float, delta_star: Optional[float] = None):
    """Compute b_reduced and b."""
    if beta.size < 2:
        b_reduced = np.zeros_like(beta)
        return b_reduced, P_c * b_reduced
    if delta_star is None or delta_star <= 0:
        delta_star = 1.0 / max(float(tau), 1e-12)
    b_reduced = np.zeros_like(beta, dtype=float)
    b_reduced[1:] = (beta[1:] - beta[:-1]) / float(delta_star)
    b_reduced[0] = b_reduced[1]
    b = P_c * b_reduced
    return b_reduced, b


def compute_agencity(
    data: Any = None,
    u: Any = None,
    *,
    xi: Any = None,
    tau=None,
    normalize: bool = True,
    normalization_method: str = "zscore",
    tau_threshold: float = 0.5,
    activity_window: int = 1,
    crm_window: int = 1,
    delta_star: Optional[float] = None,
    P_c: Any = 1.0,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AgencityResult:
    """
    Compute the canonical AgencityResult from raw data.

    Accepted inputs:
    - compute_agencity(xi, u)
    - compute_agencity({"xi": ..., "u": ...})
    - compute_agencity(AgencitySignal)
    - compute_agencity(SignalData)
    """
    preprocess_kwargs = preprocess_kwargs or {}

    if u is None:
        if hasattr(data, "xi") and hasattr(data, "u"):
            xi = np.asarray(getattr(data, "xi"), dtype=float)
            u = np.asarray(getattr(data, "u"), dtype=float)
            if metadata is None and hasattr(data, "metadata"):
                raw = getattr(data, "metadata")
                if hasattr(raw, "to_dict"):
                    metadata = raw.to_dict()
                elif isinstance(raw, dict):
                    metadata = dict(raw)
        elif isinstance(data, dict) and "xi" in data and "u" in data:
            xi = np.asarray(data["xi"], dtype=float)
            u = np.asarray(data["u"], dtype=float)
        else:
            raise ValueError("Provide either (xi, u) or a signal-like object.")
    else:
        if xi is None:
            xi = np.arange(len(u), dtype=float)
        xi = np.asarray(xi, dtype=float)
        u = np.asarray(u, dtype=float)

    if normalize or preprocess_kwargs:
        signal = prepare_signal(
            xi,
            u,
            normalize=normalize,
            normalization_method=normalization_method,
            **preprocess_kwargs,
        )
        xi, u = signal.xi, signal.u

    u_star = _normalize(u, method=normalization_method)
    X_star = _activation(u_star, xi)
    A_star = _activity(X_star, window=activity_window)
    if tau is None:
        tau = _estimate_tau(xi, X_star, threshold=tau_threshold)
    else:
        tau = float(tau)
    t_star = xi / max(float(tau), 1e-12)
    M = _memory(A_star, crm_window=crm_window)
    O = _organization(X_star, crm_window=crm_window)
    beta = _beta(X_star, A_star, M, O)
    P_c_arr = _power(P_c, xi, X_star, A_star)
    b_reduced, b = _agencity(beta, P_c_arr, tau=tau, delta_star=delta_star)

    return AgencityResult(
        xi=xi,
        u=u,
        u_star=u_star,
        X_star=X_star,
        A_star=A_star,
        tau=float(tau),
        t_star=t_star,
        M=M,
        O=O,
        beta=beta,
        b_reduced=b_reduced,
        b=b,
        P_c=P_c_arr,
        metadata=ExperimentMetadata.from_dict(metadata or {}),
        config=dict(config or {
            "normalize": normalize,
            "normalization_method": normalization_method,
            "tau_threshold": tau_threshold,
            "activity_window": activity_window,
            "crm_window": crm_window,
        }),
    )
