"""
full_pipeline.py

Advanced example using the AgencityLab pipeline.
Demonstrates end-to-end processing:
data → computation → analysis → structured insights.
"""

import numpy as np
from agencitylab.api import pipeline, visualize_agencity


def generate_signal(xi):
    """Generate a structured oscillatory signal."""
    return np.sin(xi) + 0.3 * np.cos(3 * xi)


def run_pipeline(xi, u):
    """Run the full pipeline."""
    pipe = (
        pipeline()
        .from_arrays(xi, u)
        .compute()
        .analyze()
    )

    return pipe.run()


def display_summary(context):
    """Display a clean summary of results."""
    summary = context.result.summary()
    analysis = context.artifacts["analysis"]

    print("\n=== AGENTITY ANALYSIS ===")

    print("\n--- Summary ---")
    print(
        f"Samples     : {summary['n_samples']}\n"
        f"Tau         : {summary['tau']:.4f}\n"
        f"b_mean      : {summary['b_mean']:.4f}\n"
        f"b_std       : {summary['b_std']:.4f}\n"
        f"beta range  : [{summary['beta_min']:.3f}, {summary['beta_max']:.3f}]"
    )

    print("\n--- Regime ---")
    print(f"Type        : {analysis['regime']}")

    print("\n--- Information ---")
    info = analysis["information"]
    print(
        f"Shannon H(b): {info['shannon_entropy_b']:.4f}\n"
        f"Agencity Idx: {info['agencity_information_index']:.4f}"
    )

    print("\n--- Stability ---")
    stability = analysis["stability"]
    print(
        f"Variance    : {stability['variance']:.6f}\n"
        f"Trend       : {stability['trend']:.6e}\n"
        f"Stable hint : {stability['stable_hint']}"
    )

    print("\n--- Diagnostics ---")
    diag = analysis["diagnostics"]
    print(
        f"Peak        : {diag['max']:.4f}\n"
        f"Min         : {diag['min']:.4f}\n"
        f"events   : {diag['event_count']}"
    )


def visualize(context):
    """Visualize agencity dynamics."""
    print("\nDisplaying time-series visualization...")
    visualize_agencity(context.result, kind="timeseries")


def main():
    # Coordinate (generalized time)
    xi = np.linspace(0, 20, 500)

    # Signal
    u = generate_signal(xi)

    # Run pipeline
    context = run_pipeline(xi, u)

    # Display results
    display_summary(context)

    # Visualization
    visualize(context)


if __name__ == "__main__":
    main()