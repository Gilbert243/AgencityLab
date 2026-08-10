"""Stable user-facing API namespace for AgencityLab."""

from agencitylab.models import AgencityResult, ExperimentMetadata, RESULT_SCHEMA_VERSION
from agencitylab.exceptions import (
    AgencityError,
    AgencityValidationError,
    PhysicalParameterError,
    UnitValidationError,
    BatchItemError,
    StreamStateError,
    StreamNotReadyError,
)
from agencitylab.backends.selector import get_backend
from agencitylab.analysis import ANALYSIS_SCHEMA_VERSION, RegimeCriteria

from .compute import compute_agencity
from .analyze import (
    analyze_agencity,
    textual_analysis,
    analyze_regime,
    analyze_regime_signature,
    analyze_coherence,
    analyze_geometry,
    analyze_stability,
    analyze_information,
    analyze_events,
    analyze_transitions,
    analyze_multiscale,
    analyze_signature,
)
from .pipeline_api import AgencityPipeline, pipeline
from .streaming import AgencityStream, stream_agencity
from .batch import run_batch, analyze_batch, summarize_batch, compare_batch
from .report import build_report, build_text_report, summarize, report_dict
from .export import export_json, export_csv, export_excel, export_pdf, export_report
from .visualize import visualize_agencity
from .shortcuts import run, inspect, plot, summarize as quick_summary

PipelineBuilder = AgencityPipeline
pipeline_builder = pipeline

try:
    from agencitylab.backends.selector import available_backends
except Exception:  # pragma: no cover - optional backend discovery
    def available_backends():
        return ["numpy"]


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "ANALYSIS_SCHEMA_VERSION",
    "RegimeCriteria",
    "AgencityResult",
    "ExperimentMetadata",
    "AgencityError",
    "AgencityValidationError",
    "PhysicalParameterError",
    "UnitValidationError",
    "BatchItemError",
    "StreamStateError",
    "StreamNotReadyError",
    "compute_agencity",
    "analyze_agencity",
    "textual_analysis",
    "analyze_regime",
    "analyze_regime_signature",
    "analyze_coherence",
    "analyze_geometry",
    "analyze_stability",
    "analyze_information",
    "analyze_events",
    "analyze_transitions",
    "analyze_multiscale",
    "analyze_signature",
    "pipeline",
    "AgencityPipeline",
    "PipelineBuilder",
    "pipeline_builder",
    "AgencityStream",
    "stream_agencity",
    "run_batch",
    "analyze_batch",
    "summarize_batch",
    "compare_batch",
    "build_report",
    "build_text_report",
    "summarize",
    "report_dict",
    "export_json",
    "export_csv",
    "export_excel",
    "export_pdf",
    "export_report",
    "visualize_agencity",
    "run",
    "inspect",
    "plot",
    "quick_summary",
    "get_backend",
    "available_backends",
]
