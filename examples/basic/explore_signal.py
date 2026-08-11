"""Explore diagnostics on a few deterministic/reference signals using the stable API."""

from __future__ import annotations

import numpy as np

from agencitylab import analyze_agencity, compute_agencity, visualize_agencity


def generate_signals(xi: np.ndarray) -> dict[str, np.ndarray]:
    """Generate reproducible signals for diagnostic inspection."""
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(xi))
    return {
        "Sinusoidal": np.sin(xi),
        "White noise": noise,
        "Sinus + noise": np.sin(xi) + 0.3 * noise,
        "Constant": np.ones(len(xi)),
    }


def analyze_signals(
    xi: np.ndarray,
    signals: dict[str, np.ndarray],
) -> dict[str, tuple[object, dict]]:
    """Compute Agencity and diagnostics for each signal."""
    results = {}

    for name, u in signals.items():
        result = compute_agencity(
            u=u,
            xi=xi,
            A_ref=1.0,
            tau=1.0,
            w=0.8,
            P_c=2.0,
            coordinate_unit="s",
            power_unit="W",
        )
        analysis = analyze_agencity(result)
        results[name] = (result, analysis)

        print(f"\n=== {name} ===")
        print(f"Mean |b|      : {np.mean(np.abs(result.b)):.6g}")
        print(f"Regime        : {analysis['regime']}")
        print(f"Real agencity : {analysis['real_agencity']['status']}")

    return results


def visualize(results: dict[str, tuple[object, dict]]) -> None:
    """Render stable time-series views without opening interactive windows."""
    for name, (result, _analysis) in results.items():
        print(f"Visualizing: {name}")
        visualize_agencity(result, kind="timeseries", show=False)


def main() -> None:
    # The 0.02 s sampling interval makes tau=1.0 s and w=0.8 s exact window multiples.
    xi = np.linspace(0.0, 6.0, 301)
    results = analyze_signals(xi, generate_signals(xi))
    visualize(results)


if __name__ == "__main__":
    main()
