"""
simple_compute.py

Example usage of AgencityLab:
Comparison of different signals and their agencity behavior.
"""

import numpy as np
from agencitylab.api import compute_agencity, visualize_agencity


def generate_signals(xi):
    """Generate different test signals."""
    return {
        "Sinusoidal": np.sin(xi),
        "White noise": np.random.randn(len(xi)),
        "Sinus + noise": np.sin(xi) + 0.3 * np.random.randn(len(xi)),
        "Constant": np.ones(len(xi)),
    }


def run_experiment(xi, signals):
    """Compute agencity for each signal and display results."""
    results = {}

    for name, u in signals.items():
        result = compute_agencity(xi, u)
        results[name] = result

        summary = result.summary()

        print(f"\n=== {name} ===")
        print(
            f"b_mean = {summary['b_mean']:.4f} | "
            f"b_std = {summary['b_std']:.4f} | "
            f"beta ∈ [{summary['beta_min']:.3f}, {summary['beta_max']:.3f}]"
        )

    return results


def visualize_results(results):
    """Visualize each result."""
    for name, result in results.items():
        print(f"\nDisplaying visualization for: {name}")
        visualize_agencity(result, kind="timeseries")


def main():
    # Time / coordinate
    xi = np.linspace(0, 10, 200)

    # Generate signals
    signals = generate_signals(xi)

    # Run computations
    results = run_experiment(xi, signals)

    # Visualize
    visualize_results(results)


if __name__ == "__main__":
    main()