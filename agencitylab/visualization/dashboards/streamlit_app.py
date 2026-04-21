"""
Streamlit dashboard for AgencityLab.

This module is optional and can be used as a starter dashboard.
"""

from __future__ import annotations


def run_streamlit_app(result):
    """
    Launch a very small Streamlit view of an AgencityResult.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Streamlit is required for this dashboard. Install the dashboard extra."
        ) from exc

    st.title("AgencityLab Dashboard")
    st.write("AgencityLab result summary")
    st.json(result.summary())
    st.line_chart(
        {
            "u": result.u,
            "X*": result.X_star,
            "beta": result.beta,
            "b": result.b,
        }
    )
