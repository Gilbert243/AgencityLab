"""
explore_signal.py

Explore agencity regimes and information metrics
on different types of signals.
"""

import numpy as np
from agencitylab.api import compute_agencity, analyze_agencity, visualize_agencity


def generate_signals(xi):
    """Generate test signals."""
    return {
        "Sinusoidal": np.sin(xi),
        "White noise": np.random.randn(len(xi)),
        "Sinus + noise": np.sin(xi) + 0.3 * np.random.randn(len(xi)),
        "Constant": np.ones(len(xi)),
    }


def analyze_signals(xi, signals):
    """Compute agencity + analysis for each signal."""
    results = {}

    for name, u in signals.items():
        result = compute_agencity(xi, u)
        analysis = analyze_agencity(result)

        results[name] = (result, analysis)

        entropy = analysis["information"]["shannon_entropy_b"]
        regime = analysis["regime"]

        print(f"\n=== {name} ===")
        print(f"Regime   : {regime}")
        print(f"Entropy  : {entropy:.4f}")

    return results


def visualize(results):
    """Visualize results."""
    for name, (result, _) in results.items():
        print(f"\nVisualizing: {name}")
        visualize_agencity(result, kind="timeseries")


def main():
    xi = np.linspace(0, 6.28, 300)

    signals = generate_signals(xi)
    results = analyze_signals(xi, signals)

    visualize(results)


if __name__ == "__main__":
    main()