"""
Scientific analysis entry points for AgencityLab.

This layer returns structured analysis dictionaries and text reports.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agencitylab.analysis.reports import build_report_dict, build_text_report
from agencitylab.analysis.regimes import classify_regime, detect_regime_changes
from agencitylab.analysis.stability import stability_summary
from agencitylab.analysis.diagnostics import summarize_diagnostics
from agencitylab.analysis.information import (
    agencity_information_index,
    agencity_information_density,
)
from agencitylab.analysis.information.agencity_info import full_information_summary
from agencitylab.analysis.events import detect_events, event_summary
from agencitylab.analysis.transitions import detect_transitions, transition_summary
from agencitylab.analysis.multi_scale import summarize_multiscale
from agencitylab.analysis.signature import agencity_signature


def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def analyze_agencity(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Return the full structured analysis dictionary for an AgencityResult.
    """
    if signature is None:
        signature = _get_attr(result, "signature", None)
    if multiscale is None:
        multiscale = _get_attr(result, "multiscale", None)

    if isinstance(multiscale, list):
        try:
            multiscale = summarize_multiscale(multiscale, verbose=verbose)
        except Exception:
            multiscale = None

    report = build_report_dict(result, signature=signature, multiscale=multiscale)

    if verbose:
        print("[analysis] structured report built")
        print(f"[analysis] regime = {report.get('regime', 'unknown')}")

    return report


def textual_analysis(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
) -> str:
    """Return a human-readable text report."""
    return build_text_report(result, signature=signature, multiscale=multiscale)


def analyze_regime(result, *, verbose: bool = False) -> str:
    b = _get_attr(result, "b")
    theta = _get_attr(result, "theta", None)
    regime = classify_regime(b, theta=theta, verbose=verbose)
    return regime


def analyze_stability(result, *, verbose: bool = False) -> Dict[str, Any]:
    b = _get_attr(result, "b")
    return stability_summary(b, verbose=verbose)


def analyze_information(result, *, verbose: bool = False) -> Dict[str, Any]:
    b = _get_attr(result, "b")
    info = full_information_summary(b, verbose=verbose)
    info["agencity_information_index"] = agencity_information_index(b, verbose=verbose)
    info["agencity_information_density"] = agencity_information_density(b, verbose=verbose)
    return info


def analyze_events(result, *, threshold: float = 3.0, verbose: bool = False) -> Dict[str, Any]:
    b = _get_attr(result, "b")
    return event_summary(b, threshold=threshold, component="magnitude", verbose=verbose)


def analyze_transitions(
    result,
    *,
    derivative_threshold: float = 2.0,
    window: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    b = _get_attr(result, "b")
    return transition_summary(
        b,
        derivative_threshold=derivative_threshold,
        window=window,
        component="magnitude",
        verbose=verbose,
    )


def analyze_multiscale(result, *, verbose: bool = False) -> Dict[str, Any]:
    """
    Summarize a multiscale result stored on the result object.
    """
    multiscale = _get_attr(result, "multiscale", None)
    if multiscale is None:
        return {}

    if isinstance(multiscale, dict):
        return multiscale

    if isinstance(multiscale, list):
        return summarize_multiscale(multiscale, verbose=verbose)

    return {}


def analyze_signature(result, *, verbose: bool = False) -> Dict[str, Any]:
    """
    Compute a signature from multiscale content if available.
    """
    signature = _get_attr(result, "signature", None)
    if signature is not None:
        return signature

    multiscale = analyze_multiscale(result, verbose=verbose)
    if not multiscale:
        return {}

    tau = multiscale.get("tau", None)
    beta_mean = multiscale.get("beta_mean", None)
    if tau is None or beta_mean is None:
        return {}

    return agencity_signature(tau, beta_mean, verbose=verbose)