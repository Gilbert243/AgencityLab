"""
agencity_spectrum.py

Demonstrates multi-scale agencity analysis.

Compute:
- standard agencity
- agencity spectrum b(τ)
- optimal scale detection
"""

import numpy as np

from agencitylab.api import compute_agencity, analyze_agencity
from agencitylab.analysis.multi_scale import (
    agencity_spectrum,
    print_spectrum,
    find_optimal_tau,
)


def generate_signal(xi):
    """
    Multi-frequency structured signal.
    """
    return np.sin(xi) + 0.3 * np.cos(3 * xi)


def main():
    # 🔹 Coordinate (generalized time)
    xi = np.linspace(0, 20, 500)

    # 🔹 Signal
    u = generate_signal(xi)

    # ===============================
    # 1. Single-scale analysis
    # ===============================
    result = compute_agencity(xi, u)
    analysis = analyze_agencity(result)

    summary = result.summary()

    print("\n=== SINGLE-SCALE ANALYSIS ===")
    print(
        f"Samples     : {summary['n_samples']}\n"
        f"Tau         : {summary['tau']:.4f}\n"
        f"b_mean      : {summary['b_mean']:.4f}\n"
        f"b_std       : {summary['b_std']:.4f}\n"
        f"beta range  : [{summary['beta_min']:.3f}, {summary['beta_max']:.3f}]"
    )

    print("\nRegime:", analysis["regime"])
    print(
        "Entropy:",
        f"{analysis['information']['shannon_entropy_b']:.4f}",
    )

    # ===============================
    # 2. Multi-scale spectrum
    # ===============================
    taus = [0.5, 1, 2, 5, 10]

    spectrum = agencity_spectrum(xi, u, taus)

    print_spectrum(spectrum)

    # ===============================
    # 3. Optimal scale
    # ===============================
    optimal = find_optimal_tau(xi, u, taus)

    print("\n=== OPTIMAL SCALE ===")
    print(
        f"Criterion : {optimal['criterion']}\n"
        f"Tau_opt   : {optimal['tau_opt']}\n"
        f"Value     : {optimal['value']:.4f}"
    )

    # ===============================
    # 4. Interpretation
    # ===============================
    print("\n=== INTERPRETATION ===")

    tau_opt = optimal["tau_opt"]

    if tau_opt <= 1:
        print("Dominant micro-dynamics (fast variations).")
    elif tau_opt <= 5:
        print("Intermediate structured dynamics.")
    else:
        print("Large-scale organized dynamics dominate.")


if __name__ == "__main__":
    main()