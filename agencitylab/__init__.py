"""AgencityLab package.

This package provides the scientific framework for the Agencity theory.
The public API is intentionally lightweight at import time.
"""

from .version import __version__

"""
High-level public API for AgencityLab.

This package exposes the simplest entry points for users.

Includes:
- compute
- analysis
- pipeline
- streaming
- batch processing
- reporting & export
- visualization
- shortcuts
- backend access (optional)
"""

# ============================================================
# COMPUTE
# ============================================================
from .api.compute import compute_agencity, AgencityResult

# ============================================================
# ANALYSIS
# ============================================================
from .api.analyze import (
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
from .api.pipeline_api import AgencityPipeline, pipeline

# Backward-compatible aliases
PipelineBuilder = AgencityPipeline
pipeline_builder = pipeline

# ============================================================
# STREAMING (REAL-TIME)
# ============================================================
from .api.streaming import AgencityStream, stream_agencity

# ============================================================
# BATCH (RESEARCH / MULTI SIGNALS)
# ============================================================
from .api.batch import (
    run_batch,
    analyze_batch,
    summarize_batch,
    compare_batch,
)

# ============================================================
# REPORT / EXPORT
# ============================================================
from .api.report import build_report, build_text_report, summarize
from .api.export import (
    export_json,
    export_csv,
    export_excel,
    export_pdf,
    export_report,
)

# ============================================================
# VISUALIZATION
# ============================================================
from .api.visualize import visualize_agencity

# ============================================================
# SHORTCUTS (ULTRA SIMPLE API)
# ============================================================
from .api.shortcuts import run, inspect, plot, summarize as quick_summary

# ============================================================
# BACKEND (OPTIONAL USER CONTROL)
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
    "__version__",
    # ---- compute ----
    "compute_agencity",
    "AgencityResult",

    # ---- analysis ----
    "analyze_agencity",
    "textual_analysis",
    "analyze_regime",
    "analyze_stability",
    "analyze_information",
    "analyze_events",
    "analyze_transitions",
    "analyze_multiscale",
    "analyze_signature",

    # ---- pipeline ----
    "pipeline",
    "AgencityPipeline",
    "PipelineBuilder",
    "pipeline_builder",

    # ---- streaming ----
    "AgencityStream",
    "stream_agencity",

    # ---- batch ----
    "run_batch",
    "analyze_batch",
    "summarize_batch",
    "compare_batch",

    # ---- report ----
    "build_report",
    "build_text_report",
    "summarize",

    # ---- export ----
    "export_json",
    "export_csv",
    "export_excel",
    "export_pdf",
    "export_report",

    # ---- visualization ----
    "visualize_agencity",

    # ---- shortcuts ----
    "run",
    "inspect",
    "plot",
    "quick_summary",

    # ---- backend ----
    "get_backend",
    "available_backends",
]