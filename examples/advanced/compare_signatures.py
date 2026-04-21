"""
compare_signatures.py

Compare agencity signatures across different signals.

Shows how different systems exhibit distinct multi-scale fingerprints.
"""

import numpy as np

from agencitylab.analysis.signature import (
    agencity_signature,
    print_signature,
)


def generate_signals(xi):
    """Create a diverse set of signals."""
    return {
        "Constant": np.ones_like(xi),
        "Sinus": np.sin(xi),
        "White noise": np.random.randn(len(xi)),
        "Sinus + noise": np.sin(xi) + 0.3 * np.random.randn(len(xi)),
        "Multi-scale (sin + cos)": np.sin(xi) + 0.3 * np.cos(3 * xi),
    }


def compare_signatures(xi, signals, taus):
    """Compute and display signatures for all signals."""

    results = {}

    print("\n=== MULTI-SYSTEM AGENCY SIGNATURES ===")

    for name, u in signals.items():
        sig = agencity_signature(xi, u, taus)
        results[name] = sig

        print(f"\n--- {name} ---")
        print_signature(sig)

    return results


def summary_table(results):
    """Display compact comparison table."""

    print("\n=== SUMMARY TABLE ===")
    print(f"{'Signal':20} | {'Type':18} | {'Peak τ':8} | {'Peak b_std':10}")
    print("-" * 65)

    for name, sig in results.items():
        print(
            f"{name:20} | "
            f"{sig['signature_type']:18} | "
            f"{sig['peak_tau']:8.2f} | "
            f"{sig['peak_value']:10.4f}"
        )


def main():
    # Coordinate (generalized time)
    xi = np.linspace(0, 20, 500)

    # Signals
    signals = generate_signals(xi)

    # Multi-scale exploration
    taus = [0.5, 1, 2, 5, 10, 20]

    # Compute signatures
    results = compare_signatures(xi, signals, taus)

    # Global comparison
    summary_table(results)

    # Interpretation
    print("\n=== INTERPRETATION ===")
    print(
        "Each system exhibits a distinct agencity signature.\n"
        "- Constant: no dynamics → inactive\n"
        "- Sinus: structured but low variability\n"
        "- Noise: high variability but unstable\n"
        "- Mixed: intermediate structured dynamics\n"
        "- Multi-scale: organization emerges across scales"
    )


if __name__ == "__main__":
    main()