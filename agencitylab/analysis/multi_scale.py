"""
multi_scale.py

Multi-scale analysis for AgencityLab.

Provides tools to compute the agencity spectrum:
b(τ) = agencity observed at scale τ
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Dict, Any, Optional

from agencitylab.api import compute_agencity


def agencity_spectrum(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    normalize: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute the agencity spectrum across multiple scales τ.

    Parameters
    ----------
    xi : np.ndarray
        Evolution coordinate (time, index, etc.)
    u : np.ndarray
        Observed signal
    taus : iterable of float
        List of τ values to explore
    normalize : bool
        If True, normalize b by its standard deviation

    Returns
    -------
    spectrum : list of dict
        Each entry contains:
            - tau
            - b_mean
            - b_std
            - beta_min
            - beta_max
    """

    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)

    if xi.shape != u.shape:
        raise ValueError("xi and u must have the same shape")

    spectrum = []

    for tau in taus:
        result = compute_agencity(xi, u, tau=tau)

        b = result.b
        beta = result.beta

        if normalize:
            std = np.std(b)
            if std > 1e-12:
                b = b / std

        entry = {
            "tau": float(tau),
            "b_mean": float(np.mean(b)),
            "b_std": float(np.std(b)),
            "beta_min": float(np.min(beta)),
            "beta_max": float(np.max(beta)),
        }

        spectrum.append(entry)

    return spectrum


def agencity_spectrum_array(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
) -> Dict[str, np.ndarray]:
    """
    Compute spectrum and return structured arrays (for plotting).

    Returns
    -------
    dict with:
        - tau
        - b_mean
        - b_std
    """

    spec = agencity_spectrum(xi, u, taus)

    return {
        "tau": np.array([s["tau"] for s in spec]),
        "b_mean": np.array([s["b_mean"] for s in spec]),
        "b_std": np.array([s["b_std"] for s in spec]),
    }


def find_optimal_tau(
    xi: np.ndarray,
    u: np.ndarray,
    taus: Iterable[float],
    *,
    criterion: str = "max_variability",
) -> Dict[str, Any]:
    """
    Find an optimal τ according to a criterion.

    Parameters
    ----------
    criterion : str
        "max_variability" → maximize std(b)
        "max_mean"        → maximize mean(b)

    Returns
    -------
    dict with:
        - tau_opt
        - value
        - full_spectrum
    """

    spec = agencity_spectrum(xi, u, taus)

    if criterion == "max_variability":
        key = "b_std"
    elif criterion == "max_mean":
        key = "b_mean"
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    best = max(spec, key=lambda x: x[key])

    return {
        "tau_opt": best["tau"],
        "value": best[key],
        "criterion": criterion,
        "spectrum": spec,
    }


def print_spectrum(spectrum: List[Dict[str, Any]]) -> None:
    """
    Pretty-print the spectrum.
    """

    print("\n=== AGENCY SPECTRUM ===")
    print(f"{'tau':>8} | {'b_mean':>10} | {'b_std':>10}")
    print("-" * 34)

    for s in spectrum:
        print(f"{s['tau']:8.3f} | {s['b_mean']:10.4f} | {s['b_std']:10.4f}")


def example_usage():
    """
    Example usage for quick testing.
    """

    import matplotlib.pyplot as plt

    xi = np.linspace(0, 20, 500)
    u = np.sin(xi) + 0.3 * np.cos(3 * xi)

    taus = [0.5, 1, 2, 5, 10]

    spec = agencity_spectrum(xi, u, taus)
    print_spectrum(spec)

    arr = agencity_spectrum_array(xi, u, taus)

    plt.figure()
    plt.plot(arr["tau"], arr["b_std"], marker="o")
    plt.xlabel("tau")
    plt.ylabel("b_std")
    plt.title("Agencity spectrum")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    example_usage()