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
from .analysis import ANALYSIS_SCHEMA_VERSION, RegimeCriteria
from .api.compute import compute_agencity
from .api.extensions import (
    RIEMANNIAN_EXTENSION_STATUS,
    compute_agencity_spectrum,
    compute_discrete_agencity,
    compute_multivariate_agencity,
    optimize_agencity_window,
    riemannian_extension_status,
)
from .api.analyze import (
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
from .api.pipeline_api import AgencityPipeline, pipeline
from .api.streaming import AgencityStream, stream_agencity
from .api.batch import run_batch, analyze_batch, summarize_batch, compare_batch
from .api.report import build_report, build_text_report, summarize
from .api.export import (
    SCIENTIFIC_UX_SCHEMA_VERSION,
    export_json,
    export_csv,
    export_result_csv,
    export_study_json,
    export_excel,
    export_pdf,
    export_report,
)
from .api.visualize import visualize_agencity, visualize_multiscale_spectrum
from .api.scientific import ScientificStudy, scientific_workflow
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
    "ANALYSIS_SCHEMA_VERSION",
    "SCIENTIFIC_UX_SCHEMA_VERSION",
    "RegimeCriteria",
    "AgencityResult",
    "ExperimentMetadata",
    "ScientificStudy",
    "AgencityError",
    "AgencityValidationError",
    "PhysicalParameterError",
    "UnitValidationError",
    "BatchItemError",
    "StreamStateError",
    "StreamNotReadyError",
    "compute_agencity",
    "compute_agencity_spectrum",
    "optimize_agencity_window",
    "compute_discrete_agencity",
    "compute_multivariate_agencity",
    "riemannian_extension_status",
    "RIEMANNIAN_EXTENSION_STATUS",
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
    "scientific_workflow",
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
    "export_result_csv",
    "export_study_json",
    "export_excel",
    "export_pdf",
    "export_report",
    "visualize_agencity",
    "visualize_multiscale_spectrum",
    "run",
    "inspect",
    "plot",
    "quick_summary",
    "get_backend",
    "available_backends",
]
