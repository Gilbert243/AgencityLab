"""
Scientific analysis tools for AgencityLab.

This package provides regime classification, metrics, diagnostics,
spectrum analysis and information-theoretic extensions.
"""

from .diagnostics import detect_events, summarize_diagnostics
from .metrics import (
    agencity_efficiency,
    agencity_integral,
    agencity_mean,
    agencity_peak,
    agencity_variance,
)
from .regimes import classify_regime, detect_regime_changes
from .reports import build_text_report, build_report_dict
from .spectrum import agencity_spectrum
from .stability import stability_summary

__all__ = [
    "agencity_efficiency",
    "agencity_integral",
    "agencity_mean",
    "agencity_peak",
    "agencity_spectrum",
    "agencity_variance",
    "build_report_dict",
    "build_text_report",
    "classify_regime",
    "detect_events",
    "detect_regime_changes",
    "stability_summary",
    "summarize_diagnostics",
]
