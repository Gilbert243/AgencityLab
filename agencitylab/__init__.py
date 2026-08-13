"""Small public entry point for AgencityLab.

The package top level intentionally exposes only the canonical entry point,
principal result models, scientific status, common exceptions, and a few
high-level conveniences. Specialized scientific functionality lives in
explicit namespaces such as :mod:`agencitylab.fields` and
:mod:`agencitylab.gravity`.

Top-level names published by earlier 1.1.x releases remain available lazily as
compatibility aliases. Accessing one of those deprecated locations emits a
``DeprecationWarning``; importing :mod:`agencitylab` itself emits no warning.
"""

from __future__ import annotations

from importlib import import_module
import warnings

from .api.compute import compute_agencity
from .exceptions import (
    AgencityError,
    AgencityValidationError,
    PhysicalParameterError,
    UnitValidationError,
)
from .models import (
    AgencityResult,
    DynamicalAgencityFieldSolution,
    DynamicalAgencityFieldState,
    ExperimentMetadata,
    ObservableAgencityFieldResult,
)
from .scientific_status import ScientificStatus
from .version import __version__


_LAZY_PUBLIC = {
    "analysis": ("agencitylab.analysis", None),
    "api": ("agencitylab.api", None),
    "applications": ("agencitylab.applications", None),
    "extensions": ("agencitylab.extensions", None),
    "fields": ("agencitylab.fields", None),
    "gravity": ("agencitylab.gravity", None),
    "models": ("agencitylab.models", None),
    "quantum": ("agencitylab.quantum", None),
    "thermodynamics": ("agencitylab.thermodynamics", None),
    "analyze_agencity": ("agencitylab.api.analyze", "analyze_agencity"),
    "compute_agencity_field": ("agencitylab.fields", "compute_agencity_field"),
    "scientific_workflow": ("agencitylab.api.scientific", "scientific_workflow"),
}

# Compatibility only. These names are deliberately absent from ``__all__`` so
# discovery points users to the scientific namespace that owns the concept.
_COMPAT_EXPORTS = {
    # Models and schema metadata.
    "RESULT_SCHEMA_VERSION": ("agencitylab.models", "RESULT_SCHEMA_VERSION", "agencitylab.models.RESULT_SCHEMA_VERSION"),
    "FieldModelMetadata": ("agencitylab.models", "FieldModelMetadata", "agencitylab.models.FieldModelMetadata"),
    "ParameterProvenance": ("agencitylab.models", "ParameterProvenance", "agencitylab.models.ParameterProvenance"),
    "ParameterSource": ("agencitylab.models", "ParameterSource", "agencitylab.models.ParameterSource"),
    # Analysis.
    "ANALYSIS_SCHEMA_VERSION": ("agencitylab.analysis", "ANALYSIS_SCHEMA_VERSION", "agencitylab.analysis.ANALYSIS_SCHEMA_VERSION"),
    "RegimeCriteria": ("agencitylab.analysis", "RegimeCriteria", "agencitylab.analysis.RegimeCriteria"),
    "textual_analysis": ("agencitylab.api", "textual_analysis", "agencitylab.api.textual_analysis"),
    "analyze_regime": ("agencitylab.api", "analyze_regime", "agencitylab.api.analyze_regime"),
    "analyze_regime_signature": ("agencitylab.api", "analyze_regime_signature", "agencitylab.api.analyze_regime_signature"),
    "analyze_coherence": ("agencitylab.api", "analyze_coherence", "agencitylab.api.analyze_coherence"),
    "analyze_geometry": ("agencitylab.api", "analyze_geometry", "agencitylab.api.analyze_geometry"),
    "analyze_stability": ("agencitylab.api", "analyze_stability", "agencitylab.api.analyze_stability"),
    "analyze_information": ("agencitylab.api", "analyze_information", "agencitylab.api.analyze_information"),
    "analyze_events": ("agencitylab.api", "analyze_events", "agencitylab.api.analyze_events"),
    "analyze_transitions": ("agencitylab.api", "analyze_transitions", "agencitylab.api.analyze_transitions"),
    "analyze_multiscale": ("agencitylab.api", "analyze_multiscale", "agencitylab.api.analyze_multiscale"),
    "analyze_signature": ("agencitylab.api", "analyze_signature", "agencitylab.api.analyze_signature"),
    # Stable workflow namespace: kept, but no longer advertised at package root.
    "ScientificStudy": ("agencitylab.api", "ScientificStudy", "agencitylab.api.ScientificStudy"),
    "AgencityPipeline": ("agencitylab.api", "AgencityPipeline", "agencitylab.api.AgencityPipeline"),
    "PipelineBuilder": ("agencitylab.api", "PipelineBuilder", "agencitylab.api.PipelineBuilder"),
    "pipeline": ("agencitylab.api", "pipeline", "agencitylab.api.pipeline"),
    "pipeline_builder": ("agencitylab.api", "pipeline_builder", "agencitylab.api.pipeline_builder"),
    "AgencityStream": ("agencitylab.api", "AgencityStream", "agencitylab.api.AgencityStream"),
    "stream_agencity": ("agencitylab.api", "stream_agencity", "agencitylab.api.stream_agencity"),
    "run_batch": ("agencitylab.api", "run_batch", "agencitylab.api.run_batch"),
    "analyze_batch": ("agencitylab.api", "analyze_batch", "agencitylab.api.analyze_batch"),
    "summarize_batch": ("agencitylab.api", "summarize_batch", "agencitylab.api.summarize_batch"),
    "compare_batch": ("agencitylab.api", "compare_batch", "agencitylab.api.compare_batch"),
    "build_report": ("agencitylab.api", "build_report", "agencitylab.api.build_report"),
    "build_text_report": ("agencitylab.api", "build_text_report", "agencitylab.api.build_text_report"),
    "summarize": ("agencitylab.api", "summarize", "agencitylab.api.summarize"),
    "SCIENTIFIC_UX_SCHEMA_VERSION": ("agencitylab.api", "SCIENTIFIC_UX_SCHEMA_VERSION", "agencitylab.api.SCIENTIFIC_UX_SCHEMA_VERSION"),
    "export_json": ("agencitylab.api", "export_json", "agencitylab.api.export_json"),
    "export_csv": ("agencitylab.api", "export_csv", "agencitylab.api.export_csv"),
    "export_result_csv": ("agencitylab.api", "export_result_csv", "agencitylab.api.export_result_csv"),
    "export_study_json": ("agencitylab.api", "export_study_json", "agencitylab.api.export_study_json"),
    "export_excel": ("agencitylab.api", "export_excel", "agencitylab.api.export_excel"),
    "export_pdf": ("agencitylab.api", "export_pdf", "agencitylab.api.export_pdf"),
    "export_report": ("agencitylab.api", "export_report", "agencitylab.api.export_report"),
    "visualize_agencity": ("agencitylab.api", "visualize_agencity", "agencitylab.api.visualize_agencity"),
    "visualize_multiscale_spectrum": ("agencitylab.api", "visualize_multiscale_spectrum", "agencitylab.api.visualize_multiscale_spectrum"),
    "run": ("agencitylab.api", "run", "agencitylab.api.run"),
    "inspect": ("agencitylab.api", "inspect", "agencitylab.api.inspect"),
    "plot": ("agencitylab.api", "plot", "agencitylab.api.plot"),
    "quick_summary": ("agencitylab.api", "quick_summary", "agencitylab.api.quick_summary"),
    # Extensions.
    "compute_agencity_spectrum": ("agencitylab.api", "compute_agencity_spectrum", "agencitylab.api.compute_agencity_spectrum"),
    "optimize_agencity_window": ("agencitylab.api", "optimize_agencity_window", "agencitylab.api.optimize_agencity_window"),
    "compute_discrete_agencity": ("agencitylab.api", "compute_discrete_agencity", "agencitylab.api.compute_discrete_agencity"),
    "compute_multivariate_agencity": ("agencitylab.api", "compute_multivariate_agencity", "agencitylab.api.compute_multivariate_agencity"),
    "riemannian_extension_status": ("agencitylab.api", "riemannian_extension_status", "agencitylab.api.riemannian_extension_status"),
    "RIEMANNIAN_EXTENSION_STATUS": ("agencitylab.api", "RIEMANNIAN_EXTENSION_STATUS", "agencitylab.api.RIEMANNIAN_EXTENSION_STATUS"),
    # Fields.
    "beta_to_phi": ("agencitylab.fields", "beta_to_phi", "agencitylab.fields.beta_to_phi"),
    "phi_from_observable_field": ("agencitylab.fields", "phi_from_observable_field", "agencitylab.fields.phi_from_observable_field"),
    "QuarticAgencityPotential": ("agencitylab.fields", "QuarticAgencityPotential", "agencitylab.fields.QuarticAgencityPotential"),
    "vacuum_amplitude": ("agencitylab.fields", "vacuum_amplitude", "agencitylab.fields.vacuum_amplitude"),
    "vacuum_state": ("agencitylab.fields", "vacuum_state", "agencitylab.fields.vacuum_state"),
    "field_energy_density": ("agencitylab.fields", "field_energy_density", "agencitylab.fields.field_energy_density"),
    "UniformRectilinearGrid": ("agencitylab.fields", "UniformRectilinearGrid", "agencitylab.fields.UniformRectilinearGrid"),
    "PeriodicBoundary": ("agencitylab.fields", "PeriodicBoundary", "agencitylab.fields.PeriodicBoundary"),
    "DirichletBoundary": ("agencitylab.fields", "DirichletBoundary", "agencitylab.fields.DirichletBoundary"),
    "NeumannBoundary": ("agencitylab.fields", "NeumannBoundary", "agencitylab.fields.NeumannBoundary"),
    "gradient": ("agencitylab.fields", "gradient", "agencitylab.fields.gradient"),
    "laplacian": ("agencitylab.fields", "laplacian", "agencitylab.fields.laplacian"),
    "klein_gordon_acceleration": ("agencitylab.fields", "klein_gordon_acceleration", "agencitylab.fields.klein_gordon_acceleration"),
    "dissipative_klein_gordon_acceleration": ("agencitylab.fields", "dissipative_klein_gordon_acceleration", "agencitylab.fields.dissipative_klein_gordon_acceleration"),
    "tdgl_rhs": ("agencitylab.fields", "tdgl_rhs", "agencitylab.fields.tdgl_rhs"),
    "simulate_klein_gordon": ("agencitylab.fields", "simulate_klein_gordon", "agencitylab.fields.simulate_klein_gordon"),
    "simulate_dissipative_klein_gordon": ("agencitylab.fields", "simulate_dissipative_klein_gordon", "agencitylab.fields.simulate_dissipative_klein_gordon"),
    "simulate_tdgl": ("agencitylab.fields", "simulate_tdgl", "agencitylab.fields.simulate_tdgl"),
    "domain_wall_profile": ("agencitylab.fields", "domain_wall_profile", "agencitylab.fields.domain_wall_profile"),
    "domain_wall_residual": ("agencitylab.fields", "domain_wall_residual", "agencitylab.fields.domain_wall_residual"),
    "vortex_field": ("agencitylab.fields", "vortex_field", "agencitylab.fields.vortex_field"),
    "vortex_radial_residual": ("agencitylab.fields", "vortex_radial_residual", "agencitylab.fields.vortex_radial_residual"),
    "phase_winding": ("agencitylab.fields", "phase_winding", "agencitylab.fields.phase_winding"),
    "field_zero_mask": ("agencitylab.fields", "field_zero_mask", "agencitylab.fields.field_zero_mask"),
    # Thermodynamics.
    "PhaseLawFit": ("agencitylab.thermodynamics", "PhaseLawFit", "agencitylab.thermodynamics.PhaseLawFit"),
    "contrast_agencial_entropy": ("agencitylab.thermodynamics", "contrast_agencial_entropy", "agencitylab.thermodynamics.contrast_agencial_entropy"),
    "dissipation_density": ("agencitylab.thermodynamics", "dissipation_density", "agencitylab.thermodynamics.dissipation_density"),
    "energy_balance_residual": ("agencitylab.thermodynamics", "energy_balance_residual", "agencitylab.thermodynamics.energy_balance_residual"),
    "entropy_production_density": ("agencitylab.thermodynamics", "entropy_production_density", "agencitylab.thermodynamics.entropy_production_density"),
    "field_agencial_entropy": ("agencitylab.thermodynamics", "field_agencial_entropy", "agencitylab.thermodynamics.field_agencial_entropy"),
    "landauer_agencity_power": ("agencitylab.thermodynamics", "landauer_agencity_power", "agencitylab.thermodynamics.landauer_agencity_power"),
    "landauer_characteristic_power": ("agencitylab.thermodynamics", "landauer_characteristic_power", "agencitylab.thermodynamics.landauer_characteristic_power"),
    "modulus_law_margin": ("agencitylab.thermodynamics", "modulus_law_margin", "agencitylab.thermodynamics.modulus_law_margin"),
    "modulus_law_satisfied": ("agencitylab.thermodynamics", "modulus_law_satisfied", "agencitylab.thermodynamics.modulus_law_satisfied"),
    "phase_law_prediction": ("agencitylab.thermodynamics", "phase_law_prediction", "agencitylab.thermodynamics.phase_law_prediction"),
    "phase_law_residual": ("agencitylab.thermodynamics", "phase_law_residual", "agencitylab.thermodynamics.phase_law_residual"),
    "second_law_residual": ("agencitylab.thermodynamics", "second_law_residual", "agencitylab.thermodynamics.second_law_residual"),
    "structural_information_rate": ("agencitylab.thermodynamics", "structural_information_rate", "agencitylab.thermodynamics.structural_information_rate"),
    "temperature_dependent_lambda": ("agencitylab.thermodynamics", "temperature_dependent_lambda", "agencitylab.thermodynamics.temperature_dependent_lambda"),
    "thermal_reference_phase_fit": ("agencitylab.thermodynamics", "thermal_reference_phase_fit", "agencitylab.thermodynamics.thermal_reference_phase_fit"),
    # Gravity.
    "GRAVITY_METRIC_SIGNATURE": ("agencitylab.gravity", "GRAVITY_METRIC_SIGNATURE", "agencitylab.gravity.GRAVITY_METRIC_SIGNATURE"),
    "conformal_coupling": ("agencitylab.gravity", "conformal_coupling", "agencitylab.gravity.conformal_coupling"),
    "covariant_scalar_derivative": ("agencitylab.gravity", "covariant_scalar_derivative", "agencitylab.gravity.covariant_scalar_derivative"),
    "curved_field_residual": ("agencitylab.gravity", "curved_field_residual", "agencitylab.gravity.curved_field_residual"),
    "einstein_equation_residual": ("agencitylab.gravity", "einstein_equation_residual", "agencitylab.gravity.einstein_equation_residual"),
    "matter_lagrangian_density": ("agencitylab.gravity", "matter_lagrangian_density", "agencitylab.gravity.matter_lagrangian_density"),
    "minimal_coupling": ("agencitylab.gravity", "minimal_coupling", "agencitylab.gravity.minimal_coupling"),
    "minkowski_inverse_metric": ("agencitylab.gravity", "minkowski_inverse_metric", "agencitylab.gravity.minkowski_inverse_metric"),
    "minkowski_metric": ("agencitylab.gravity", "minkowski_metric", "agencitylab.gravity.minkowski_metric"),
    "nonminimal_coupling_density": ("agencitylab.gravity", "nonminimal_coupling_density", "agencitylab.gravity.nonminimal_coupling_density"),
    "stress_energy_tensor": ("agencitylab.gravity", "stress_energy_tensor", "agencitylab.gravity.stress_energy_tensor"),
    # Quantum and cosmology.
    "agencity_uncertainty_lower_bound": ("agencitylab.quantum", "agencity_uncertainty_lower_bound", "agencitylab.quantum.agencity_uncertainty_lower_bound"),
    "annihilation_operator": ("agencitylab.quantum", "annihilation_operator", "agencitylab.quantum.annihilation_operator"),
    "one_loop_quartic_beta": ("agencitylab.quantum", "one_loop_quartic_beta", "agencitylab.quantum.one_loop_quartic_beta"),
    "radial_angular_frequency": ("agencitylab.quantum", "radial_angular_frequency", "agencitylab.quantum.radial_angular_frequency"),
    "radial_mass_squared": ("agencitylab.quantum", "radial_mass_squared", "agencitylab.quantum.radial_mass_squared"),
    "FlatFLRWSolution": ("agencitylab.applications.cosmology", "FlatFLRWSolution", "agencitylab.applications.cosmology.FlatFLRWSolution"),
    "friedmann_constraint_residual": ("agencitylab.applications.cosmology", "friedmann_constraint_residual", "agencitylab.applications.cosmology.friedmann_constraint_residual"),
    "homogeneous_energy_density": ("agencitylab.applications.cosmology", "homogeneous_energy_density", "agencitylab.applications.cosmology.homogeneous_energy_density"),
    "homogeneous_pressure": ("agencitylab.applications.cosmology", "homogeneous_pressure", "agencitylab.applications.cosmology.homogeneous_pressure"),
    "simulate_flat_flrw": ("agencitylab.applications.cosmology", "simulate_flat_flrw", "agencitylab.applications.cosmology.simulate_flat_flrw"),
    # Backend helpers.
    "get_backend": ("agencitylab.backends.selector", "get_backend", "agencitylab.backends.selector.get_backend"),
    "available_backends": ("agencitylab.backends.selector", "available_backends", "agencitylab.backends.selector.available_backends"),
    "backend_capabilities": ("agencitylab.backends.selector", "backend_capabilities", "agencitylab.backends.selector.backend_capabilities"),
    # Specialized exceptions.
    "BatchItemError": ("agencitylab.exceptions", "BatchItemError", "agencitylab.exceptions.BatchItemError"),
    "StreamStateError": ("agencitylab.exceptions", "StreamStateError", "agencitylab.exceptions.StreamStateError"),
    "StreamNotReadyError": ("agencitylab.exceptions", "StreamNotReadyError", "agencitylab.exceptions.StreamNotReadyError"),
}


def _resolve(module_name: str, attribute: str | None):
    module = import_module(module_name)
    return module if attribute is None else getattr(module, attribute)


def __getattr__(name: str):
    public = _LAZY_PUBLIC.get(name)
    if public is not None:
        value = _resolve(*public)
        globals()[name] = value
        return value

    compatibility = _COMPAT_EXPORTS.get(name)
    if compatibility is not None:
        module_name, attribute, replacement = compatibility
        warnings.warn(
            f"agencitylab.{name} is deprecated at package top level since 1.1.7; "
            f"use {replacement} instead. The compatibility alias is retained for "
            "the 1.x line and may be removed in a future major release.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = _resolve(module_name, attribute)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'agencitylab' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_PUBLIC) | set(_COMPAT_EXPORTS))


__all__ = [
    "__version__",
    "ScientificStatus",
    "AgencityResult",
    "ExperimentMetadata",
    "ObservableAgencityFieldResult",
    "DynamicalAgencityFieldState",
    "DynamicalAgencityFieldSolution",
    "AgencityError",
    "AgencityValidationError",
    "PhysicalParameterError",
    "UnitValidationError",
    "compute_agencity",
    "compute_agencity_field",
    "analyze_agencity",
    "scientific_workflow",
    "analysis",
    "fields",
    "thermodynamics",
    "gravity",
    "quantum",
    "applications",
]
