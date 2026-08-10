"""High-level public API for AgencityLab."""

from .version import __version__
from .models import AgencityResult, ExperimentMetadata, RESULT_SCHEMA_VERSION
from .exceptions import (
    AgencityError,
    AgencityValidationError,
    PhysicalParameterError,
    UnitValidationError,
    BatchItemError,
    StreamStateError,
    StreamNotReadyError,
)
from .api.compute import compute_agencity
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
from .api.pipeline_api import AgencityPipeline, pipeline
from .api.streaming import AgencityStream, stream_agencity
from .api.batch import run_batch, analyze_batch, summarize_batch, compare_batch
from .api.report import build_report, build_text_report, summarize
from .api.export import export_json, export_csv, export_excel, export_pdf, export_report
from .api.visualize import visualize_agencity
from .api.shortcuts import run, inspect, plot, summarize as quick_summary
from .backends.selector import get_backend

PipelineBuilder = AgencityPipeline
pipeline_builder = pipeline

try:
    from .backends.selector import available_backends
except Exception:  # pragma: no cover - optional backend discovery
    def available_backends():
        return ["numpy"]


__all__ = [
    "__version__",
    "RESULT_SCHEMA_VERSION",
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
