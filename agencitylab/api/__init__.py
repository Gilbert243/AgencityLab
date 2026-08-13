"""Stable user-facing orchestration namespace for AgencityLab 1.0."""

from agencitylab.analysis import ANALYSIS_SCHEMA_VERSION, RegimeCriteria
from agencitylab.backends.selector import available_backends, get_backend
from agencitylab.exceptions import (
    AgencityError,
    AgencityValidationError,
    BatchItemError,
    PhysicalParameterError,
    StreamNotReadyError,
    StreamStateError,
    UnitValidationError,
)
from agencitylab.models import AgencityResult, ExperimentMetadata, RESULT_SCHEMA_VERSION

from .analyze import (
    analyze_agencity,
    analyze_coherence,
    analyze_events,
    analyze_geometry,
    analyze_information,
    analyze_multiscale,
    analyze_regime,
    analyze_regime_signature,
    analyze_signature,
    analyze_stability,
    analyze_transitions,
    textual_analysis,
)
from .batch import analyze_batch, run_batch, summarize_batch
from .compute import compute_agencity
from .export import (
    SCIENTIFIC_UX_SCHEMA_VERSION,
    export_csv,
    export_excel,
    export_json,
    export_pdf,
    export_report,
    export_result_csv,
    export_study_json,
)
from .extensions import (
    RIEMANNIAN_EXTENSION_STATUS,
    compute_agencity_spectrum,
    compute_discrete_agencity,
    compute_multivariate_agencity,
    optimize_agencity_window,
    riemannian_extension_status,
)
from .pipeline_api import AgencityPipeline, pipeline
from .report import build_report, build_text_report, report_dict, summarize
from .scientific import ScientificStudy, scientific_workflow
from .streaming import AgencityStream, stream_agencity
from .visualize import visualize_agencity, visualize_multiscale_spectrum

__all__ = [
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
    "AgencityStream",
    "stream_agencity",
    "run_batch",
    "analyze_batch",
    "summarize_batch",
    "build_report",
    "build_text_report",
    "summarize",
    "report_dict",
    "export_json",
    "export_csv",
    "export_result_csv",
    "export_study_json",
    "export_excel",
    "export_pdf",
    "export_report",
    "visualize_agencity",
    "visualize_multiscale_spectrum",
    "get_backend",
    "available_backends",
]
