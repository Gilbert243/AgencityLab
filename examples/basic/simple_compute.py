"""Minimal reproducible AgencityLab computations on simple reference signals."""

from __future__ import annotations

import numpy as np

from agencitylab import analyze_agencity, compute_agencity


def reference_signals(xi: np.ndarray) -> dict[str, np.ndarray]:
    """Return deterministic signals useful for first-pass inspection."""
    return {
        "rest": np.ones_like(xi),
        "sinusoid": np.sin(xi),
        "damped_oscillator": np.exp(-0.05 * xi) * np.sin(xi),
    }


def main() -> None:
    xi = np.linspace(0.0, 30.0, 1201)

    for name, u in reference_signals(xi).items():
        result = compute_agencity(
            u=u,
            xi=xi,
            A_ref=1.0,
            tau=2.0,
            w=1.5,
            P_c=5.0,
            coordinate_unit="s",
            power_unit="W",
        )
        analysis = analyze_agencity(result)

        print(f"\n=== {name} ===")
        print(f"samples       : {len(result)}")
        print(f"tau, w        : {result.tau:g}, {result.memory_window:g}")
        print(f"mean |b|      : {np.mean(np.abs(result.b)):.6g} {result.b_unit}")
        print(f"mean J        : {np.mean(result.J):.6g}")
        print(f"regime        : {analysis['regime']}")
        print(f"real agencity : {analysis['real_agencity']['status']}")


if __name__ == "__main__":
    main()
