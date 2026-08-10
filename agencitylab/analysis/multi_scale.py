"""
Multi-scale analysis for AgencityLab.

Computes agencity across multiple characteristic times tau.
This is the analysis-side interpretation of the scale dependence.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

import numpy as np

from agencitylab.core.safeguards import EPS
from agencitylab.core.validation import validate_axis, validate_signal
from agencitylab.core.normalization import normalize_signal
from agencitylab.core.activation import compute_activation
from agencitylab.core.activity import compute_activity
from agencitylab.core.memory import memory
from agencitylab.core.organization import organization
from agencitylab.core.intensity import compute_intensities
from agencitylab.core.beta import compute_beta
from agencitylab.core.power import estimate_characteristic_power
from agencitylab.core.agencity import agencity


def _default_scales(n=12, low=0.5, high=2.5):
    """Log-spaced multiplicative factors."""
    return np.exp(np.linspace(np.log(low), np.log(high), n))


def _compute_at_tau(
    xi: np.ndarray,
    u: np.ndarray,
    tau: float,
    *,
    normalize: bool = False,
    verbose: bool = False,
):
    """
    Compute the full agencity pipeline at a fixed tau.
    """
    u_star, _ = normalize_signal(u)
    X = compute_activation(u_star, xi)
    A = compute_activity(X, xi)

    M = memory(A, tau, axis=xi, verbose=verbose)
    O = organization(X, tau, axis=xi, verbose=verbose)

    D, S = compute_intensities(X, A, M, O, verbose=verbose)
    J, U, beta = compute_beta(D, S, M, O, verbose=verbose)

    P_c = estimate_characteristic_power(u, tau=tau, verbose=verbose)
    b = agencity(beta, P_c, verbose=verbose)

    if normalize:
        s = np.std(np.abs(b))
        if s > EPS:
            b = b / s

    return {
        "tau": float(tau),
        "u_star": u_star,
        "X": X,
        "A": A,
        "M": M,
        "O": O,
        "D": D,
        "S": S,
        "J": J,
        "U": U,
        "beta": beta,
        "P_c": P_c,
        "b": b,
    }


def agencity_multiscale(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    normalize: bool = False,
    return_full: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute the agencity spectrum across multiple scales tau.

    Returns a list of dictionaries, one per tau.
    """
    xi = validate_axis(xi)
    u = validate_signal(u).ravel()

    if xi.shape != u.shape:
        raise ValueError("xi and u must have the same shape")

    spectrum = []

    for tau in taus:
        if verbose:
            print(f"[multiscale] tau = {float(tau):.6g}")

        res = _compute_at_tau(xi, u, float(tau), normalize=normalize, verbose=verbose)

        b = res["b"]
        beta = res["beta"]

        entry = {
            "tau": float(tau),
            "b_mean": float(np.mean(np.abs(b))),
            "b_std": float(np.std(np.abs(b))),
            "beta_mean": float(np.mean(np.abs(beta))),
            "beta_std": float(np.std(np.abs(beta))),
            "J_mean": float(np.mean(res["J"])),
            "P_c": float(np.asarray(res["P_c"]).mean()),
        }

        if return_full:
            entry["raw"] = res

        spectrum.append(entry)

    return spectrum


def agencity_spectrum_array(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    normalize: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute the multiscale spectrum and return arrays.
    """
    spec = agencity_multiscale(
        xi, u, taus, normalize=normalize, return_full=False, verbose=verbose
    )

    return {
        "tau": np.array([s["tau"] for s in spec], dtype=float),
        "b_mean": np.array([s["b_mean"] for s in spec], dtype=float),
        "b_std": np.array([s["b_std"] for s in spec], dtype=float),
        "beta_mean": np.array([s["beta_mean"] for s in spec], dtype=float),
        "beta_std": np.array([s["beta_std"] for s in spec], dtype=float),
        "J_mean": np.array([s["J_mean"] for s in spec], dtype=float),
        "P_c": np.array([s["P_c"] for s in spec], dtype=float),
    }


def find_optimal_tau(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    criterion: str = "max_beta_mean",
    normalize: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Find an optimal tau according to a criterion.

    Supported criteria:
        - max_beta_mean
        - max_b_mean
        - max_J_mean
        - max_variability
        - min_variability
    """
    spec = agencity_multiscale(
        xi, u, taus, normalize=normalize, return_full=False, verbose=verbose
    )

    if criterion == "max_beta_mean":
        key = "beta_mean"
        best = max(spec, key=lambda x: x[key])
    elif criterion == "max_b_mean":
        key = "b_mean"
        best = max(spec, key=lambda x: x[key])
    elif criterion == "max_J_mean":
        key = "J_mean"
        best = max(spec, key=lambda x: x[key])
    elif criterion == "max_variability":
        key = "b_std"
        best = max(spec, key=lambda x: x[key])
    elif criterion == "min_variability":
        key = "b_std"
        best = min(spec, key=lambda x: x[key])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return {
        "tau_opt": best["tau"],
        "value": best[key],
        "criterion": criterion,
        "spectrum": spec,
    }


def summarize_multiscale(spectrum: List[Dict[str, Any]], *, verbose: bool = False):
    """
    Return a concise interpretation of the tau spectrum.
    """
    if not spectrum:
        return {}

    tau = np.array([s["tau"] for s in spectrum], dtype=float)
    beta_mean = np.array([s["beta_mean"] for s in spectrum], dtype=float)
    b_mean = np.array([s["b_mean"] for s in spectrum], dtype=float)

    idx_beta = int(np.argmax(beta_mean))
    idx_b = int(np.argmax(b_mean))

    trend = "decreasing" if beta_mean[0] > beta_mean[-1] else "increasing"

    out = {
        "tau_opt_beta": float(tau[idx_beta]),
        "tau_opt_b": float(tau[idx_b]),
        "beta_peak": float(beta_mean[idx_beta]),
        "b_peak": float(b_mean[idx_b]),
        "trend": trend,
        "n_scales": int(len(spectrum)),
    }

    if verbose:
        print("[multiscale] ---")
        for k, v in out.items():
            print(f"[multiscale] {k}: {v}")

    return out


def print_spectrum(spectrum: List[Dict[str, Any]]) -> None:
    """
    Pretty-print the spectrum.
    """
    print("\n=== AGENCITY SPECTRUM ===")
    print(f"{'tau':>8} | {'beta_mean':>10} | {'b_mean':>10} | {'b_std':>10}")
    print("-" * 48)

    for s in spectrum:
        print(
            f"{s['tau']:8.3f} | {s['beta_mean']:10.4f} | "
            f"{s['b_mean']:10.4f} | {s['b_std']:10.4f}"
        )


# Backward-compatible alias
agencity_spectrum = agencity_multiscale