"""
User-facing public API for AgencityLab.

This package exposes the main entry points for end users:
- compute
- analysis
- pipeline
- batch
- reporting / export
- visualization
- shortcuts
- backend selection helpers (optional)
"""

# ============================================================
# COMPUTE
# ============================================================

from .compute import compute_agencity, AgencityResult

# ============================================================
# ANALYSIS
# ============================================================

from .analyze import (
    analyze_agencity,
    textual_analysis,
    analyze_regime,
    analyze_stability,
    analyze_information,
    analyze_events,
    analyze_transitions,
    analyze_multiscale,
    analyze_signature,
)

# ============================================================
# PIPELINE
# ============================================================

from .pipeline_api import AgencityPipeline, pipeline

# Backward-compatible aliases
PipelineBuilder = AgencityPipeline
pipeline_builder = pipeline

# ============================================================
# STREAMING
# ============================================================

from .streaming import AgencityStream, stream_agencity

# ============================================================
# BATCH
# ============================================================

from .batch import (
    run_batch,
    analyze_batch,
    summarize_batch,
    compare_batch,
)

# ============================================================
# REPORT / EXPORT
# ============================================================

from .report import build_report, build_text_report, summarize, report_dict

from .export import (
    export_json,
    export_csv,
    export_excel,
    export_pdf,
    export_report,
)

# ============================================================
# VISUALIZATION
# ============================================================

from .visualize import visualize_agencity

# ============================================================
# SHORTCUTS
# ============================================================

from .shortcuts import run, inspect, plot, summarize as quick_summary

# ============================================================
# BACKEND
# ============================================================

from agencitylab.backends.selector import get_backend

try:
    from agencitylab.backends.selector import available_backends
except Exception:  # fallback safe
    def available_backends():
        return ["numpy"]

# ============================================================
# PUBLIC EXPORT
# ============================================================

__all__ = [
    # compute
    "compute_agencity",
    "AgencityResult",

    # analysis
    "analyze_agencity",
    "textual_analysis",
    "analyze_regime",
    "analyze_stability",
    "analyze_information",
    "analyze_events",
    "analyze_transitions",
    "analyze_multiscale",
    "analyze_signature",

    # pipeline
    "pipeline",
    "AgencityPipeline",
    "PipelineBuilder",
    "pipeline_builder",

    # streaming
    "AgencityStream",
    "stream_agencity",

    # batch
    "run_batch",
    "analyze_batch",
    "summarize_batch",
    "compare_batch",

    # report
    "build_report",
    "build_text_report",
    "summarize",
    "report_dict",

    # export
    "export_json",
    "export_csv",
    "export_excel",
    "export_pdf",
    "export_report",

    # visualization
    "visualize_agencity",

    # shortcuts
    "run",
    "inspect",
    "plot",
    "quick_summary",

    # backend
    "get_backend",
    "available_backends",
]