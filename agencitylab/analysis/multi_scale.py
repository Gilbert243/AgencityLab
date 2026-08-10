"""Analysis-side summaries for the v0.6 multiscale extension.

All numerical Agencity equations are delegated to :mod:`agencitylab.core.multiscale`.
This module only reshapes and interprets the resulting spectra.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from agencitylab.core.multiscale import agencity_spectrum as _core_spectrum


def _mean_power(P_c, xi) -> float:
    """Return a descriptive mean of an already supplied characteristic power."""
    candidate = P_c(np.asarray(xi, dtype=float)) if callable(P_c) else P_c
    return float(np.mean(np.asarray(candidate, dtype=float)))


def agencity_multiscale(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    A_ref=None,
    P_c=None,
    windows=None,
    normalize: bool = False,
    return_full: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Return one summary dictionary per explicitly supplied structural scale.

    ``A_ref`` and ``P_c`` are required physical/contextual quantities. The old
    analysis helper inferred them from the observed signal; v0.6 intentionally
    rejects that behaviour. ``normalize=True`` is also rejected because spectrum
    normalization would alter the physical magnitude of ``b``.

    Historical descriptive keys such as ``b_std`` and ``beta_std`` are retained
    for compatibility. They remain diagnostics and do not alter the spectrum.
    """
    del verbose
    if normalize:
        raise ValueError("multiscale b must not be signal-normalized; normalize only diagnostic plots")
    if A_ref is None:
        raise ValueError("A_ref must be supplied explicitly for multiscale analysis")
    if P_c is None:
        raise ValueError("P_c must be supplied explicitly for multiscale analysis")

    spectrum = _core_spectrum(
        xi,
        u,
        taus,
        A_ref=A_ref,
        P_c=P_c,
        windows=windows,
        return_full=return_full,
    )
    power_mean = _mean_power(P_c, xi)

    entries: List[Dict[str, Any]] = []
    for index, tau in enumerate(spectrum["tau"]):
        b_abs = np.abs(spectrum["b"][index])
        beta_abs = np.abs(spectrum["beta"][index])
        entry: Dict[str, Any] = {
            "tau": float(tau),
            "w": float(spectrum["w"][index]),
            "b_mean": float(spectrum["b_mean"][index]),
            "b_rms": float(spectrum["b_rms"][index]),
            "b_std": float(np.std(b_abs)),
            "beta_mean": float(spectrum["beta_mean"][index]),
            "beta_std": float(np.std(beta_abs)),
            "J_mean": float(spectrum["J_mean"][index]),
            "S_mean": float(spectrum["S_mean"][index]),
            "P_c": power_mean,
            "window_mode": spectrum["window_mode"],
        }
        if return_full:
            entry["raw"] = spectrum["responses"][index]
        entries.append(entry)
    return entries


def agencity_spectrum_array(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    A_ref=None,
    P_c=None,
    windows=None,
    normalize: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Return scalar summary arrays of the ``b(t, tau)`` spectrum."""
    entries = agencity_multiscale(
        xi,
        u,
        taus,
        A_ref=A_ref,
        P_c=P_c,
        windows=windows,
        normalize=normalize,
        return_full=False,
        verbose=verbose,
    )
    keys = (
        "tau",
        "w",
        "b_mean",
        "b_rms",
        "b_std",
        "beta_mean",
        "beta_std",
        "J_mean",
        "S_mean",
        "P_c",
    )
    return {key: np.asarray([entry[key] for entry in entries], dtype=float) for key in keys}


def find_optimal_tau(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    A_ref=None,
    P_c=None,
    windows=None,
    criterion: str = "max_beta_mean",
    normalize: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Select a diagnostic scale from an explicitly defined spectrum.

    This function does **not** estimate the physical characteristic time ``tau``.
    It only selects one supplied scale according to a stated descriptive metric.
    Window optimisation is a different operation implemented by
    ``optimize_agencity_window`` in the public API.

    Historical ``max_variability`` and ``min_variability`` retain their previous
    meaning based on the standard deviation of ``|b|``.
    """
    entries = agencity_multiscale(
        xi,
        u,
        taus,
        A_ref=A_ref,
        P_c=P_c,
        windows=windows,
        normalize=normalize,
        return_full=False,
        verbose=verbose,
    )
    selectors = {
        "max_beta_mean": ("beta_mean", np.argmax),
        "max_b_mean": ("b_mean", np.argmax),
        "max_J_mean": ("J_mean", np.argmax),
        "max_variability": ("b_std", np.argmax),
        "min_variability": ("b_std", np.argmin),
    }
    if criterion not in selectors:
        raise ValueError(f"Unknown criterion: {criterion}")
    key, selector = selectors[criterion]
    values = np.asarray([entry[key] for entry in entries], dtype=float)
    index = int(selector(values))
    return {
        "tau_opt": entries[index]["tau"],
        "w": entries[index]["w"],
        "value": float(values[index]),
        "criterion": criterion,
        "status": "diagnostic scale selection; not physical tau inference",
        "spectrum": entries,
    }


def summarize_multiscale(spectrum: List[Dict[str, Any]], *, verbose: bool = False):
    """Return a concise descriptive summary of an already computed spectrum."""
    if not spectrum:
        return {}
    tau = np.asarray([item["tau"] for item in spectrum], dtype=float)
    beta_mean = np.asarray([item["beta_mean"] for item in spectrum], dtype=float)
    b_mean = np.asarray([item["b_mean"] for item in spectrum], dtype=float)
    idx_beta = int(np.argmax(beta_mean))
    idx_b = int(np.argmax(b_mean))
    trend = "decreasing" if beta_mean[0] > beta_mean[-1] else "increasing"
    out = {
        "tau_peak_beta": float(tau[idx_beta]),
        "tau_peak_b": float(tau[idx_b]),
        # Compatibility aliases retained from the pre-v0.6 analysis API.
        "tau_opt_beta": float(tau[idx_beta]),
        "tau_opt_b": float(tau[idx_b]),
        "beta_peak": float(beta_mean[idx_beta]),
        "b_peak": float(b_mean[idx_b]),
        "trend": trend,
        "n_scales": int(tau.size),
        "status": "descriptive multiscale summary; peaks are not physical tau inference",
    }
    if verbose:
        for key, value in out.items():
            print(f"[multiscale] {key}: {value}")
    return out


def print_spectrum(spectrum: List[Dict[str, Any]]) -> None:
    """Print a compact human-readable multiscale summary."""
    print("\n=== AGENCITY SPECTRUM ===")
    print(f"{'tau':>10} | {'w':>10} | {'beta_mean':>12} | {'b_mean':>12}")
    print("-" * 56)
    for item in spectrum:
        print(
            f"{item['tau']:10.4g} | {item['w']:10.4g} | "
            f"{item['beta_mean']:12.5g} | {item['b_mean']:12.5g}"
        )


agencity_spectrum = agencity_multiscale
