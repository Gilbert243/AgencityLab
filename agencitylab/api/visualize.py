"""
Visualization entry points for AgencityLab.
"""

from __future__ import annotations


def visualize_agencity(result, kind: str = "timeseries", show: bool = True, **kwargs):
    """
    Create a visualization for an AgencityResult.

    Supported kinds:
    - timeseries
    - components
    - phase
    - spectrum
    """
    try:
        from agencitylab.visualization.timeseries import plot_timeseries
        from agencitylab.visualization.components import plot_components
        from agencitylab.visualization.phase import plot_phase_portrait
        from agencitylab.visualization.spectrum import plot_spectrum
    except Exception as exc:
        raise ImportError(
            "Visualization package is not available in the current build."
        ) from exc

    kind = kind.lower().strip()

    if kind not in {"timeseries", "components", "phase", "spectrum"}:
        raise ValueError(
            f"Unknown visualization kind '{kind}'. "
            "Available: timeseries, components, phase, spectrum"
        )

    if kind == "timeseries":
        return plot_timeseries(result, show=show, **kwargs)

    if kind == "components":
        return plot_components(result, show=show, **kwargs)

    if kind == "phase":
        return plot_phase_portrait(result, show=show, **kwargs)

    if kind == "spectrum":
        return plot_spectrum(result, show=show, **kwargs)