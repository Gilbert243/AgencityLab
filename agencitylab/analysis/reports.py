"""Structured scientific reporting for the Agencity analysis layer."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .anomalies import anomaly_summary
from .coherence import (
    detect_structural_plateaus,
    full_coherence,
    orientation_statistics,
    phase_coherence,
    real_agencity_criterion,
    sigma_theta,
)
from .complexity import complexity_summary
from .correlation import full_correlation_summary
from .diagnostics import summarize_diagnostics
from .events import dynamic_peak_summary
from .geometry import geometric_summary
from .information import agencity_information_density, agencity_information_index
from .information.agencity_info import full_information_summary
from .metrics import (
    agencity_components,
    agencity_energy,
    agencity_mean,
    agencity_peak,
    agencity_variance,
    global_efficiency,
)
from .regimes import classify_regime, regime_signature
from .stability import stability_summary
from .transitions import (
    critical_surface_crossings,
    detect_theta_jumps,
    zero_summary,
)

ANALYSIS_SCHEMA_VERSION = "0.5"


def _portable(value):
    """Convert analysis output to JSON-friendly values without losing metadata."""
    if is_dataclass(value):
        return _portable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def _memory_window(result) -> float:
    """Return the CRM width, falling back to the historical w=tau convention."""
    window = getattr(result, "memory_window", None)
    return float(result.tau) if window is None else float(window)


def _analysis_start(result) -> int:
    """Return the finite-record index after two complete CRM windows."""
    xi = np.asarray(result.xi, dtype=float)
    return int(np.searchsorted(xi, xi[0] + 2.0 * _memory_window(result), side="left"))


def _real_agencity(result, sigma, thresholds: Mapping[str, Any] | None):
    kwargs = dict(thresholds or {})
    return real_agencity_criterion(result.S, sigma, result.b, **kwargs)


def build_report_dict(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
    real_agencity_thresholds: Mapping[str, Any] | None = None,
    regime_criteria=None,
    theta_jump_threshold: float | None = None,
    plateau_slope_threshold: float | None = None,
    plateau_min_duration: float | None = None,
    d_peak_prominence: float | None = None,
) -> Dict[str, Any]:
    """Build a scientific analysis report from an already-computed result.

    Nothing in this function recomputes or modifies ``X*``, ``A*``, CRM, ``M``,
    ``O``, ``D``, ``S``, ``J``, ``Theta``, ``beta`` or ``b``. Threshold-bearing
    operations are explicit diagnostics and are recorded in the report.
    """
    xi = np.asarray(result.xi, dtype=float)
    beta = np.asarray(result.beta, dtype=complex)
    b = np.asarray(result.b, dtype=complex)
    S = np.asarray(result.S, dtype=float)
    D = np.asarray(result.D, dtype=float)
    J = np.asarray(result.J, dtype=float)
    theta = np.asarray(result.theta, dtype=float)
    P_c = getattr(result, "P_c", None)

    structural = S > 0.0
    sigma = sigma_theta(theta, xi, result.tau, valid_mask=structural)
    finite_sigma = np.isfinite(sigma)
    start = _analysis_start(result)

    xi_valid = xi[start:]
    beta_valid = beta[start:]
    theta_valid = theta[start:]
    S_valid = S[start:]
    D_valid = D[start:]
    J_valid = J[start:]
    structural_valid = S_valid > 0.0

    raw_signature = regime_signature(result)
    regime = classify_regime(raw_signature, criteria=regime_criteria)
    real_diag = _real_agencity(result, sigma, real_agencity_thresholds)

    geometry = geometric_summary(
        beta_valid,
        xi=xi_valid,
        theta=theta_valid,
        valid_mask=structural_valid,
    ) if beta_valid.size else {
        "geometry_source": "beta",
        "curvature": [],
        "curvature_mean": float("nan"),
        "curvature_mean_abs": float("nan"),
        "curvature_std": float("nan"),
        "curvature_defined_fraction": 0.0,
        "winding": {
            "defined": False,
            "winding_number": float("nan"),
            "nearest_integer": 0,
            "integer_residual": float("nan"),
        },
    }

    zeros = zero_summary(S_valid, J_valid, xi_valid, atol=0.0)
    zeros["indices"] = np.asarray(zeros["indices"], dtype=int) + start
    crossings = critical_surface_crossings(D_valid, S_valid)
    crossing_indices = crossings + start
    crossing_times = xi_valid[crossings] if crossings.size else np.asarray([], dtype=float)

    if theta_jump_threshold is None:
        theta_jumps = {
            "status": "not configured",
            "threshold": None,
            "indices": [],
            "times": [],
        }
    else:
        jump_indices = detect_theta_jumps(
            theta_valid,
            threshold=theta_jump_threshold,
            valid_mask=structural_valid,
        )
        theta_jumps = {
            "status": "diagnostic threshold configured",
            "threshold": float(theta_jump_threshold),
            "indices": jump_indices + start,
            "times": xi_valid[jump_indices] if jump_indices.size else np.asarray([], dtype=float),
        }

    if plateau_slope_threshold is None and plateau_min_duration is None:
        plateaus = {
            "status": "not configured",
            "slope_threshold": None,
            "min_duration": None,
            "intervals": [],
        }
    elif plateau_slope_threshold is None or plateau_min_duration is None:
        raise ValueError("plateau_slope_threshold and plateau_min_duration must be provided together")
    else:
        plateaus = {
            "status": "diagnostic thresholds configured",
            "slope_threshold": float(plateau_slope_threshold),
            "min_duration": float(plateau_min_duration),
            "intervals": detect_structural_plateaus(
                S,
                xi,
                slope_threshold=plateau_slope_threshold,
                min_duration=plateau_min_duration,
            ),
        }

    orientation = orientation_statistics(result.M, result.O)
    orientation["sigma_theta_mean"] = (
        float(np.mean(sigma[finite_sigma])) if np.any(finite_sigma) else float("nan")
    )
    orientation["sigma_theta"] = sigma
    orientation["structural_phase_coherence"] = phase_coherence(
        theta,
        valid_mask=structural,
        values_are_angles=True,
    )
    orientation["sigma_theta_definition"] = "Var(Theta(s); s in [t-tau,t])"

    bx, by, mag, phase = agencity_components(b)
    summary = result.summary() if hasattr(result, "summary") else {}
    metadata = (
        result.metadata.to_dict()
        if hasattr(result, "metadata") and hasattr(result.metadata, "to_dict")
        else {}
    )

    report: Dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "scientific_boundary": (
            "analysis/diagnostics only; theoretical computation is consumed without redefinition"
        ),
        "metadata": metadata,
        "summary": summary,
        "components": {
            "bx_mean": float(np.mean(bx)),
            "by_mean": float(np.mean(by)),
            "magnitude_mean": float(np.mean(mag)),
            "phase_mean": float(np.mean(phase)),
        },
        "metrics": {
            "mean_magnitude": agencity_mean(b, component="magnitude"),
            "variance_magnitude": agencity_variance(b, component="magnitude"),
            "peak": agencity_peak(b),
            "energy": agencity_energy(b),
            "efficiency": global_efficiency(b, P_c) if P_c is not None else None,
            "J_mean": float(np.mean(J)),
            "beta_mean": float(np.mean(np.abs(beta))),
        },
        "regime": regime,
        "regime_signature": raw_signature,
        "regime_classification": {
            "label": regime,
            "criteria": regime_criteria,
            "status": (
                "configured diagnostic"
                if regime_criteria is not None
                else "undetermined without contextual criteria"
            ),
        },
        "real_agencity": real_diag,
        "coherence": {
            "structural_orientation": orientation,
            "flux_diagnostics": full_coherence(b),
        },
        "geometry": geometry,
        "events": {
            "dynamic_peaks": dynamic_peak_summary(D, xi, prominence=d_peak_prominence),
        },
        "transitions": {
            "zeros": zeros,
            "critical_surface_D_equals_S": {
                "indices": crossing_indices,
                "times": crossing_times,
                "definition": "D = S iff J = 0",
            },
            "theta_jumps": theta_jumps,
        },
        "structural_plateaus": plateaus,
        "analysis_interval": {
            "finite_record_crm_start_index": int(start),
            "finite_record_crm_start_time": float(xi[start]) if start < xi.size else None,
            "memory_window": _memory_window(result),
            "rule": "t >= t0 + 2*w for CRM-dependent finite-record geometry/transitions",
        },
        # Compatibility diagnostics retained from earlier versions. They are not
        # used to alter theoretical outputs or the theory-facing classifier.
        "stability": stability_summary(b),
        "information": {
            "entropy": agencity_information_index(b),
            "density": agencity_information_density(b),
            **full_information_summary(b),
        },
        "complexity": complexity_summary(b),
        "correlation": full_correlation_summary(b, beta=beta, J=J),
        "anomalies": anomaly_summary(b),
        "diagnostics": {
            **summarize_diagnostics(b, theta=theta),
            "status": "legacy descriptive compatibility",
        },
    }

    if signature is not None:
        report["signature"] = signature
    if multiscale is not None:
        report["multiscale"] = multiscale

    return _portable(report)


def build_text_report(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
    **analysis_kwargs,
) -> str:
    """Build a concise researcher-facing report from the structured analysis."""
    report = build_report_dict(
        result,
        signature=signature,
        multiscale=multiscale,
        **analysis_kwargs,
    )
    s = report.get("summary", {})
    m = report.get("metrics", {})
    coherence = report.get("coherence", {}).get("structural_orientation", {})
    real_diag = report.get("real_agencity", {})
    geometry = report.get("geometry", {})
    winding = geometry.get("winding", {})
    coordinate_unit = getattr(result, "coordinate_unit", "") or ""
    b_unit = getattr(result, "b_unit", "") or ""
    window = _memory_window(result)

    lines = [
        "AgencityLab Scientific Analysis",
        "=" * 45,
        "",
        f"Samples                  : {s.get('n_samples', len(result.xi))}",
        f"Tau                      : {float(result.tau):.6g} {coordinate_unit}".rstrip(),
        f"CRM window w             : {window:.6g} {coordinate_unit}".rstrip(),
        f"A_ref                    : {float(result.A_ref):.6g} {getattr(result, 'unit', '')}".rstrip(),
        f"Mean |b|                 : {m.get('mean_magnitude', float('nan')):.6g} {b_unit}".rstrip(),
        f"Mean J                   : {m.get('J_mean', float('nan')):.6g}",
        f"Sigma_Theta mean         : {coherence.get('sigma_theta_mean', float('nan')):.6g}",
        f"Mean |kappa|             : {geometry.get('curvature_mean_abs', float('nan')):.6g}",
        f"Winding                  : {winding.get('winding_number', float('nan'))}",
        f"Regime                   : {report.get('regime', 'undetermined')}",
        f"Real-agencity status     : {real_diag.get('status', 'undetermined')}",
        f"Real-agencity fraction   : {real_diag.get('real_agencity_fraction', float('nan'))}",
        "",
        "Interpretation thresholds are contextual diagnostics, not theory constants.",
    ]

    if "signature" in report:
        sig = report["signature"]
        lines.extend(
            [
                "",
                "Multiscale signature",
                f"Slope alpha              : {sig.get('slope', float('nan'))}",
            ]
        )
    return "\n".join(lines)
