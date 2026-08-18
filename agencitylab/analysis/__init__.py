"""Scientific interpretation layer for AgencityLab.

The analysis package consumes canonical outputs produced by ``agencitylab.core``
and ``compute_agencity``. It may expose diagnostics, thresholds, classifications,
and summaries, but it must never redefine the canonical equations.
"""

from .metrics import (
    agencity_components,
    agencity_mean,
    agencity_variance,
    agencity_peak,
    agencity_integral,
    agencity_energy,
    squared_flux_norm,
    agencity_power_mean,
    agencity_efficiency,
    global_efficiency,
)
from .regimes import (
    RegimeCriteria,
    classify_regime,
    detect_regime_changes,
    regime_signature,
)
from .stability import stability_summary
from .information import (
    shannon_entropy,
    conditional_entropy,
    agencity_information_index,
    agencity_information_density,
    agencity_structural_information,
    agencity_phase_information,
    full_information_summary,
    landauer_lower_bound,
)
from .events import (
    detect_dynamic_peaks,
    dynamic_peak_summary,
    detect_events,
    event_summary,
)
from .transitions import (
    detect_agencity_zeros,
    critical_surface_crossings,
    detect_theta_jumps,
    zero_summary,
    detect_transitions,
    transition_summary,
)
from .diagnostics import (
    compute_energy,
    compute_derivative,
    rolling_variance,
    summarize_diagnostics,
)
from .geometry import (
    compute_angle,
    trajectory_length,
    curvature,
    radius,
    net_phase_turns,
    winding_number,
    winding_diagnostic,
    geometric_summary,
)
from .inverse import recoverable_agencity_signature
from .robustness import (
    logarithmic_contrast_offset_sensitivity,
    multiplicative_power_perturbation,
)
from .correlation import full_correlation_summary
from .coherence import (
    sigma_theta,
    phase_coherence,
    amplitude_coherence,
    temporal_coherence,
    orientation_statistics,
    angular_stability,
    detect_structural_plateaus,
    real_agencity_criterion,
    full_coherence,
    full_structural_coherence,
)
from .complexity import complexity_summary
from .anomalies import anomaly_summary
from .multi_scale import agencity_multiscale, agencity_spectrum_array, find_optimal_tau
from .signature import agencity_signature
from .reports import ANALYSIS_SCHEMA_VERSION, build_report_dict, build_text_report
from .validity import AnalysisInterval, analysis_valid_mask, resolve_analysis_interval

__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "agencity_components",
    "agencity_mean",
    "agencity_variance",
    "agencity_peak",
    "agencity_integral",
    "agencity_energy",
    "squared_flux_norm",
    "agencity_power_mean",
    "agencity_efficiency",
    "global_efficiency",
    "RegimeCriteria",
    "classify_regime",
    "detect_regime_changes",
    "regime_signature",
    "stability_summary",
    "shannon_entropy",
    "conditional_entropy",
    "agencity_information_index",
    "agencity_information_density",
    "agencity_structural_information",
    "agencity_phase_information",
    "full_information_summary",
    "landauer_lower_bound",
    "detect_dynamic_peaks",
    "dynamic_peak_summary",
    "detect_events",
    "event_summary",
    "detect_agencity_zeros",
    "critical_surface_crossings",
    "detect_theta_jumps",
    "zero_summary",
    "detect_transitions",
    "transition_summary",
    "compute_energy",
    "compute_derivative",
    "rolling_variance",
    "summarize_diagnostics",
    "compute_angle",
    "trajectory_length",
    "curvature",
    "radius",
    "net_phase_turns",
    "winding_number",
    "winding_diagnostic",
    "geometric_summary",
    "recoverable_agencity_signature",
    "logarithmic_contrast_offset_sensitivity",
    "multiplicative_power_perturbation",
    "full_correlation_summary",
    "sigma_theta",
    "phase_coherence",
    "amplitude_coherence",
    "temporal_coherence",
    "orientation_statistics",
    "angular_stability",
    "detect_structural_plateaus",
    "real_agencity_criterion",
    "full_coherence",
    "full_structural_coherence",
    "complexity_summary",
    "anomaly_summary",
    "agencity_multiscale",
    "agencity_spectrum_array",
    "find_optimal_tau",
    "agencity_signature",
    "build_report_dict",
    "build_text_report",
    "AnalysisInterval",
    "analysis_valid_mask",
    "resolve_analysis_interval",
]
