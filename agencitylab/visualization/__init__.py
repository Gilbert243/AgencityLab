"""
Visualization layer for AgencityLab.

This package provides a lightweight plotting interface for the canonical
AgencityResult object.
"""

from .components import plot_components
from .export import export_figure, save_figure
from .heatmaps import plot_heatmap
from .phase import plot_phase_portrait
from .spectrum import plot_spectrum
from .styles import apply_default_style, set_matplotlib_style
from .timeseries import plot_timeseries

__all__ = [
    "apply_default_style",
    "export_figure",
    "plot_components",
    "plot_heatmap",
    "plot_phase_portrait",
    "plot_spectrum",
    "plot_timeseries",
    "save_figure",
    "set_matplotlib_style",
]
