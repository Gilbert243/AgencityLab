"""
Visualization entry points for AgencityLab.
"""

from __future__ import annotations

import importlib


def _import_plotter(module_name: str, func_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def visualize_agencity(result, kind: str = "timeseries", show: bool = True, **kwargs):
    """
    Create a visualization for an AgencityResult.

    Supported kinds:
    - timeseries
    - components
    - phase
    - spectrum
    - heatmap
    - causal
    - attractor
    - attractor3d
    """
    kind = str(kind).lower().strip()

    mapping = {
        "timeseries": ("agencitylab.visualization.timeseries", "plot_timeseries"),
        "components": ("agencitylab.visualization.components", "plot_components"),
        "phase": ("agencitylab.visualization.phase", "plot_phase_portrait"),
        "spectrum": ("agencitylab.visualization.spectrum", "plot_spectrum"),
        "heatmap": ("agencitylab.visualization.heatmaps", "plot_heatmap"),
        "causal": ("agencitylab.visualization.causal", "plot_causal_mo"),
        "attractor": ("agencitylab.visualization.attractor", "plot_attractor"),
        "attractor3d": ("agencitylab.visualization.attractor", "plot_attractor_3d"),
    }

    if kind not in mapping:
        raise ValueError(
            f"Unknown visualization kind '{kind}'. "
            f"Available: {', '.join(sorted(mapping.keys()))}"
        )

    module_name, func_name = mapping[kind]

    try:
        plotter = _import_plotter(module_name, func_name)
    except Exception as exc:
        raise ImportError(
            "Visualization package is not available in the current build."
        ) from exc

    return plotter(result, show=show, **kwargs)