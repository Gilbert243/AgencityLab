"""
User-facing reporting API for AgencityLab.

This module exposes simple report entry points for end users,
while delegating scientific logic to agencitylab.analysis.reports.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from agencitylab.analysis.reports import build_report_dict as _build_report_dict
from agencitylab.analysis.reports import build_text_report as _build_text_report


def build_report(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Return the structured scientific report dictionary.
    """
    return _build_report_dict(result, signature=signature, multiscale=multiscale)


def build_text_report(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
) -> str:
    """
    Return a human-readable scientific report.
    """
    return _build_text_report(result, signature=signature, multiscale=multiscale)


def summarize(result) -> str:
    """
    Alias kept for user convenience.
    """
    return build_text_report(result)


def report_dict(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Alias returning the structured dict.
    """
    return build_report(result, signature=signature, multiscale=multiscale)