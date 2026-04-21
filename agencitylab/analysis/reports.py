"""
Report generation utilities for AgencityLab.
"""

from __future__ import annotations

from typing import Dict, Any

from .diagnostics import summarize_diagnostics
from .regimes import classify_regime
from .stability import stability_summary
from .metrics import agencity_mean, agencity_peak, agencity_variance


def build_report_dict(result) -> Dict[str, Any]:
    """
    Build a structured report dictionary from an AgencityResult-like object.
    """
    b = result.b

    return {
        "metadata": result.metadata.to_dict() if hasattr(result.metadata, "to_dict") else {},
        "summary": result.summary() if hasattr(result, "summary") else {},
        "metrics": {
            "mean": agencity_mean(b),
            "variance": agencity_variance(b),
            "peak": agencity_peak(b),
        },
        "regime": classify_regime(b),
        "stability": stability_summary(b),
        "diagnostics": summarize_diagnostics(b),
    }


def build_text_report(result) -> str:
    """
    Build a human-readable text report.
    """
    report = build_report_dict(result)
    summary = report["summary"]
    stability = report["stability"]
    diagnostics = report["diagnostics"]

    lines = [
        "AgencityLab Report",
        "=" * 18,
        f"Regime: {report['regime']}",
        f"Samples: {summary.get('n_samples', 0)}",
        f"Tau: {summary.get('tau', 0.0):.6g}",
        f"Mean b: {report['metrics']['mean']:.6g}",
        f"Variance b: {report['metrics']['variance']:.6g}",
        f"Peak |b|: {report['metrics']['peak']:.6g}",
        f"Trend: {stability['trend']:.6g}",
        f"events: {diagnostics['event_count']}",
    ]
    return "\n".join(lines)
