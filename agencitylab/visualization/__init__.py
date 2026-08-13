"""Scientific visualization layer for AgencityLab."""

from .components import plot_components
from .export import export_figure, save_figure
from .heatmaps import plot_heatmap
from .phase import plot_phase_portrait
from .scientific import (
    plot_beta_geometry,
    plot_multiscale_spectrum,
    plot_scientific_diagnostics,
    plot_scientific_overview,
)
from .spectrum import plot_spectrum
from .styles import apply_default_style, set_matplotlib_style
from .timeseries import plot_timeseries

__all__ = [
    "apply_default_style",
    "export_figure",
    "plot_beta_geometry",
    "plot_components",
    "plot_heatmap",
    "plot_multiscale_spectrum",
    "plot_phase_portrait",
    "plot_scientific_diagnostics",
    "plot_scientific_overview",
    "plot_spectrum",
    "plot_timeseries",
    "save_figure",
    "set_matplotlib_style",
]
