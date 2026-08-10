"""
User-friendly shortcuts for AgencityLab.

Ultra-simple API for quick usage:
- run()
- summarize()
- inspect()
- plot()
- stream()
- batch()
"""

from __future__ import annotations

from typing import Any, Iterable

from .compute import compute_agencity
from .analyze import analyze_agencity, textual_analysis
from .visualize import visualize_agencity
from .batch import run_batch
from .streaming import AgencityStream


# ============================================================
# BASIC
# ============================================================

def run(data=None, u=None, xi=None, **kwargs):
    """
    Compute Agencity and return result.

    Example:
        result = run(u=my_signal)
    """
    return compute_agencity(data=data, u=u, xi=xi, **kwargs)


def summarize(result) -> str:
    """
    Quick text summary.
    """
    return textual_analysis(result)


def inspect(result, **kwargs):
    """
    Return full analysis dict.
    """
    return analyze_agencity(result, **kwargs)


def plot(result, kind: str = "timeseries", show: bool = True, **kwargs):
    """
    Quick visualization.
    """
    return visualize_agencity(result, kind=kind, show=show, **kwargs)


# ============================================================
# STREAMING
# ============================================================

def stream(window_size=None, **kwargs):
    """
    Create a streaming processor.

    Example:
        s = stream(window_size=500)
        s.update(chunk)
    """
    return AgencityStream(window_size=window_size, **kwargs)


# ============================================================
# BATCH
# ============================================================

def batch(items: Iterable[Any], *, analyze: bool = False, **kwargs):
    """
    Run batch computation.

    Example:
        results = batch([u1, u2, u3])
    """
    return run_batch(items, analyze=analyze, **kwargs)


# ============================================================
# EXPORT SHORTCUT
# ============================================================

def export(result, path: str, *, format=None):
    """
    Quick export helper.

    Example:
        export(result, "report.json")
    """
    from .report import build_report
    from .export import export_report

    report = build_report(result)
    return export_report(report, path, format=format)