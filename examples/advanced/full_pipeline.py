"""
full_pipeline.py

Clean + enhanced AgencityLab pipeline
"""

import numpy as np
from agencitylab.api import pipeline, visualize_agencity


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(xi):
    """Structured oscillatory signal (multi-scale)"""
    return (
        np.sin(xi)
        + 0.3 * np.cos(3 * xi)
        + 0.1 * np.sin(10 * xi)
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(xi, u):

    pipe = (
        pipeline()
        .from_arrays(xi, u)
        .normalize("zscore")       # 🔥 important
        .detrend("linear")         # 🔥 important
        .smooth(window_size=5)     # 🔥 stabilise
        .compute()
        .analyze()
    )

    return pipe.run()


# ============================================================
# SUMMARY
# ============================================================

def display_summary(context):

    summary = context.result.summary()
    analysis = context.artifacts["analysis"]

    print("\n=== AGENTITY ANALYSIS ===")

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k:12}: {v}")

    print("\n--- Regime ---")
    print(analysis["regime"])

    print("\n--- Information ---")
    for k, v in analysis["information"].items():
        print(f"{k:20}: {v}")

    print("\n--- Stability ---")
    for k, v in analysis["stability"].items():
        print(f"{k:20}: {v}")

    print("\n--- Diagnostics ---")
    for k, v in analysis["diagnostics"].items():
        print(f"{k:20}: {v}")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize(context):

    print("\n📊 Time-series")
    visualize_agencity(context.result, kind="timeseries")

    print("\n📊 Components")
    visualize_agencity(context.result, kind="components")

    print("\n📊 Phase")
    visualize_agencity(context.result, kind="phase")

    print("\n📊 Spectrum")
    visualize_agencity(context.result, kind="spectrum")


# ============================================================
# MAIN
# ============================================================

def main():

    xi = np.linspace(0, 20, 500)
    u = generate_signal(xi)

    context = run_pipeline(xi, u)

    display_summary(context)
    visualize(context)


if __name__ == "__main__":
    main()