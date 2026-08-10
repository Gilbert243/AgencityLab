"""Scientific analysis entry points for AgencityLab.

The API consumes :class:`AgencityResult` objects produced by the canonical
compute layer. Analysis may apply explicit diagnostic thresholds, but it never
recomputes or changes the canonical equations.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from agencitylab.analysis.coherence import (
    orientation_statistics,
    phase_coherence,
    real_agencity_criterion,
    sigma_theta,
)
from agencitylab.analysis.events import dynamic_peak_summary, event_summary
from agencitylab.analysis.geometry import geometric_summary
from agencitylab.analysis.information import (
    agencity_information_density,
    agencity_information_index,
)
from agencitylab.analysis.information.agencity_info import full_information_summary
from agencitylab.analysis.multi_scale import summarize_multiscale
from agencitylab.analysis.regimes import classify_regime, regime_signature
from agencitylab.analysis.reports import build_report_dict, build_text_report
from agencitylab.analysis.signature import agencity_signature
from agencitylab.analysis.stability import stability_summary
from agencitylab.analysis.transitions import (
    critical_surface_crossings,
    detect_theta_jumps,
    transition_summary,
    zero_summary,
)


def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _warmup_start(result) -> int:
    xi = np.asarray(_get_attr(result, "xi"), dtype=float)
    tau = float(_get_attr(result, "tau"))
    return int(np.searchsorted(xi, xi[0] + 2.0 * tau, side="left"))


def analyze_agencity(
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
    verbose: bool = False,
) -> Dict[str, Any]:
    """Return the complete v0.5 scientific analysis dictionary.

    Threshold-bearing arguments are diagnostics. Omitting them leaves the
    corresponding interpretation unconfigured rather than inventing universal
    constants for real agencity or regime classification.
    """
    if signature is None:
        signature = _get_attr(result, "signature", None)
    if multiscale is None:
        multiscale = _get_attr(result, "multiscale", None)

    if isinstance(multiscale, list):
        try:
            multiscale = summarize_multiscale(multiscale, verbose=verbose)
        except Exception:
            multiscale = None

    report = build_report_dict(
        result,
        signature=signature,
        multiscale=multiscale,
        real_agencity_thresholds=real_agencity_thresholds,
        regime_criteria=regime_criteria,
        theta_jump_threshold=theta_jump_threshold,
        plateau_slope_threshold=plateau_slope_threshold,
        plateau_min_duration=plateau_min_duration,
        d_peak_prominence=d_peak_prominence,
    )
    if verbose:
        print("[analysis] v0.5 structured report built")
        print(f"[analysis] regime = {report.get('regime', 'undetermined')}")
    return report


def textual_analysis(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
    **analysis_kwargs,
) -> str:
    """Return a human-readable v0.5 analysis report."""
    return build_text_report(
        result,
        signature=signature,
        multiscale=multiscale,
        **analysis_kwargs,
    )


def analyze_regime(result, *, criteria=None, verbose: bool = False) -> str:
    """Classify a regime only when contextual criteria are explicitly supplied."""
    return classify_regime(result, criteria=criteria, verbose=verbose)


def analyze_coherence(
    result,
    *,
    real_agencity_thresholds: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return theory-facing structural coherence and real-agencity diagnostics."""
    M = np.asarray(_get_attr(result, "M"), dtype=float)
    O = np.asarray(_get_attr(result, "O"), dtype=float)
    S = np.asarray(_get_attr(result, "S"), dtype=float)
    theta = np.asarray(_get_attr(result, "theta"), dtype=float)
    xi = np.asarray(_get_attr(result, "xi"), dtype=float)
    tau = float(_get_attr(result, "tau"))
    b = np.asarray(_get_attr(result, "b"), dtype=complex)
    valid = S > 0.0
    sigma = sigma_theta(theta, xi, tau, valid_mask=valid)
    orientation = orientation_statistics(M, O)
    orientation["structural_phase_coherence"] = phase_coherence(
        theta,
        valid_mask=valid,
        values_are_angles=True,
    )
    orientation["sigma_theta"] = sigma
    finite = np.isfinite(sigma)
    orientation["sigma_theta_mean"] = (
        float(np.mean(sigma[finite])) if np.any(finite) else float("nan")
    )
    real_diag = real_agencity_criterion(
        S,
        sigma,
        b,
        **dict(real_agencity_thresholds or {}),
    )
    return {
        "orientation": orientation,
        "real_agencity": real_diag,
        "scientific_boundary": "diagnostic layer; canonical Theta, S, and b are unchanged",
    }


def analyze_geometry(result) -> Dict[str, Any]:
    """Return curvature and winding diagnostics of the intrinsic beta trajectory."""
    xi = np.asarray(_get_attr(result, "xi"), dtype=float)
    beta = np.asarray(_get_attr(result, "beta"), dtype=complex)
    theta = np.asarray(_get_attr(result, "theta"), dtype=float)
    S = np.asarray(_get_attr(result, "S"), dtype=float)
    start = _warmup_start(result)
    return geometric_summary(
        beta[start:],
        xi=xi[start:],
        theta=theta[start:],
        valid_mask=S[start:] > 0.0,
    )


def analyze_stability(result, *, verbose: bool = False) -> Dict[str, Any]:
    """Return the historical descriptive stability summary for compatibility."""
    b = _get_attr(result, "b")
    return stability_summary(b, verbose=verbose)


def analyze_information(result, *, verbose: bool = False) -> Dict[str, Any]:
    b = _get_attr(result, "b")
    info = full_information_summary(b, verbose=verbose)
    info["agencity_information_index"] = agencity_information_index(b, verbose=verbose)
    info["agencity_information_density"] = agencity_information_density(b, verbose=verbose)
    return info


def analyze_events(
    result,
    *,
    d_peak_prominence: float | None = None,
    threshold: float = 3.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Return D peaks while preserving the pre-v0.5 b-outlier summary keys."""
    D = _get_attr(result, "D")
    xi = _get_attr(result, "xi")
    b = _get_attr(result, "b")
    legacy = event_summary(
        b,
        threshold=threshold,
        component="magnitude",
        verbose=verbose,
    )
    return {
        **legacy,
        "legacy_b_outliers": legacy,
        "dynamic_peaks": dynamic_peak_summary(D, xi, prominence=d_peak_prominence),
    }


def analyze_transitions(
    result,
    *,
    theta_jump_threshold: float | None = None,
    derivative_threshold: float = 2.0,
    window: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Return exact theory transitions while preserving pre-v0.5 summary keys."""
    xi = np.asarray(_get_attr(result, "xi"), dtype=float)
    S = np.asarray(_get_attr(result, "S"), dtype=float)
    D = np.asarray(_get_attr(result, "D"), dtype=float)
    J = np.asarray(_get_attr(result, "J"), dtype=float)
    theta = np.asarray(_get_attr(result, "theta"), dtype=float)
    b = _get_attr(result, "b")
    start = _warmup_start(result)

    zeros = zero_summary(S[start:], J[start:], xi[start:], atol=0.0)
    zero_indices = np.asarray(zeros["indices"], dtype=int) + start
    zeros["indices"] = zero_indices.tolist()
    crossings = critical_surface_crossings(D[start:], S[start:])
    crossing_indices = crossings + start

    if theta_jump_threshold is None:
        jumps = {
            "status": "not configured",
            "threshold": None,
            "indices": [],
            "times": [],
        }
    else:
        relative = detect_theta_jumps(
            theta[start:],
            threshold=theta_jump_threshold,
            valid_mask=S[start:] > 0.0,
        )
        jumps = {
            "status": "diagnostic threshold configured",
            "threshold": float(theta_jump_threshold),
            "indices": (relative + start).tolist(),
            "times": xi[start:][relative].tolist(),
        }

    legacy = transition_summary(
        b,
        derivative_threshold=derivative_threshold,
        window=window,
        component="magnitude",
        verbose=verbose,
    )
    return {
        **legacy,
        "legacy_b_transition_heuristic": legacy,
        "zeros": zeros,
        "critical_surface_D_equals_S": {
            "indices": crossing_indices.tolist(),
            "times": xi[start:][crossings].tolist(),
            "definition": "D = S iff J = 0",
        },
        "theta_jumps": jumps,
    }


def analyze_multiscale(result, *, verbose: bool = False) -> Dict[str, Any]:
    """Summarize a multiscale result stored on the result object."""
    multiscale = _get_attr(result, "multiscale", None)
    if multiscale is None:
        return {}
    if isinstance(multiscale, dict):
        return multiscale
    if isinstance(multiscale, list):
        return summarize_multiscale(multiscale, verbose=verbose)
    return {}


def analyze_signature(
    result,
    *,
    slope_threshold: float | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute an explicitly diagnostic multiscale scaling signature."""
    existing = _get_attr(result, "signature", None)
    if existing is not None and slope_threshold is None:
        return existing

    multiscale = analyze_multiscale(result, verbose=verbose)
    if not multiscale:
        return {}
    tau = multiscale.get("tau", None)
    beta_mean = multiscale.get("beta_mean", None)
    if tau is None or beta_mean is None:
        return {}
    return agencity_signature(
        tau,
        beta_mean,
        slope_threshold=slope_threshold,
        verbose=verbose,
    )


def analyze_regime_signature(result) -> Dict[str, Any]:
    """Return the threshold-free single-scale regime signature."""
    return regime_signature(result)
