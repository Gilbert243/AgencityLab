"""Research-facing visualization entry points."""

from __future__ import annotations


def visualize_agencity(result, kind: str = "overview", show: bool = True, **kwargs):
    """Create a scientific figure from an :class:`AgencityResult`.

    Recommended v0.7 views are ``overview``, ``geometry`` and ``diagnostics``.
    Historical plot kinds remain available for compatibility. Diagnostic plots
    require an already-computed ``analysis`` dictionary and never invent
    thresholds or classifications.
    """
    kind = str(kind).strip().lower()

    try:
        if kind == "overview":
            from agencitylab.visualization.scientific import plot_scientific_overview

            return plot_scientific_overview(result, show=show, **kwargs)
        if kind == "geometry":
            from agencitylab.visualization.scientific import plot_beta_geometry

            return plot_beta_geometry(result, show=show, **kwargs)
        if kind == "diagnostics":
            from agencitylab.visualization.scientific import plot_scientific_diagnostics

            return plot_scientific_diagnostics(result, show=show, **kwargs)
        if kind == "timeseries":
            from agencitylab.visualization.timeseries import plot_timeseries

            return plot_timeseries(result, show=show, **kwargs)
        if kind == "components":
            from agencitylab.visualization.components import plot_components

            return plot_components(result, show=show, **kwargs)
        if kind in {"phase", "beta_phase"}:
            from agencitylab.visualization.phase import plot_phase_portrait

            return plot_phase_portrait(result, show=show, **kwargs)
        if kind in {"spectrum", "frequency_spectrum"}:
            from agencitylab.visualization.spectrum import plot_spectrum

            return plot_spectrum(result, show=show, **kwargs)
        if kind == "heatmap":
            from agencitylab.visualization.heatmaps import plot_heatmap

            return plot_heatmap(result, show=show, **kwargs)
        if kind == "causal":
            from agencitylab.visualization.causal import plot_causal_mo

            return plot_causal_mo(result, show=show, **kwargs)
        if kind == "attractor":
            from agencitylab.visualization.attractor import plot_attractor

            return plot_attractor(result, show=show, **kwargs)
        if kind == "attractor3d":
            from agencitylab.visualization.attractor import plot_attractor_3d

            return plot_attractor_3d(result, show=show, **kwargs)
    except ImportError as exc:
        raise ImportError("visualization requires the optional 'viz' dependencies") from exc

    supported = (
        "overview, geometry, diagnostics, timeseries, components, phase, "
        "frequency_spectrum, heatmap, causal, attractor, attractor3d"
    )
    raise ValueError(f"unknown visualization kind '{kind}'; choose from: {supported}")


def visualize_multiscale_spectrum(spectrum, *, show: bool = True, **kwargs):
    """Plot a ``compute_agencity_spectrum`` result as ``|b(t, tau)|``."""
    try:
        from agencitylab.visualization.scientific import plot_multiscale_spectrum
    except ImportError as exc:
        raise ImportError("visualization requires the optional 'viz' dependencies") from exc
    return plot_multiscale_spectrum(spectrum, show=show, **kwargs)
