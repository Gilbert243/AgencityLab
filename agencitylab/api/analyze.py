"""
Scientific analysis entry points for AgencityLab.
"""

from __future__ import annotations

from typing import Any, Dict

from agencitylab.analysis import (
    build_report_dict,
    build_text_report,
    classify_regime,
    detect_events,
    detect_regime_changes,
)
from agencitylab.analysis.information import (
    agencity_information_index,
    shannon_entropy,
)


def analyze_agencity(result) -> Dict[str, Any]:
    """
    Return a structured analysis dictionary for an AgencityResult.
    """
    report = build_report_dict(result)
    report["information"] = {
        "agencity_information_index": agencity_information_index(result.b),
        "shannon_entropy_b": shannon_entropy(abs(result.b)),
        "regime": classify_regime(result.b),
        "event_indices": detect_events(result.b).tolist(),
        "regime_changes": detect_regime_changes(result.b),
    }
    return report


def textual_analysis(result) -> str:
    """Return a human-readable text report."""
    return build_text_report(result)
