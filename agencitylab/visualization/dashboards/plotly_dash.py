"""
Plotly dashboard helpers for AgencityLab.

Plotly is an optional dependency. This module only loads it when needed.
"""

from __future__ import annotations


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
        return go
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Plotly is required for interactive dashboards. Install the dashboard extra."
        ) from exc


def build_plotly_figure(result):
    """
    Build a compact interactive Plotly figure for the main Agencity observables.
    """
    go = _require_plotly()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.xi, y=result.u, mode="lines", name="u"))
    fig.add_trace(go.Scatter(x=result.xi, y=result.X_star, mode="lines", name="X*"))
    fig.add_trace(go.Scatter(x=result.xi, y=result.beta, mode="lines", name="beta"))
    fig.add_trace(go.Scatter(x=result.xi, y=result.b, mode="lines", name="b"))
    fig.update_layout(
        title="AgencityLab interactive view",
        xaxis_title="xi",
        yaxis_title="value",
        template="plotly_white",
    )
    return fig
