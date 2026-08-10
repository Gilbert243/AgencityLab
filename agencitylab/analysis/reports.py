"""
Scientific reporting utilities for AgencityLab.

This module builds structured scientific reports from computed agencity results.
It is the SINGLE source of truth for analytical reporting.

No formatting for UI here — only structured data.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from agencitylab.core import decompose_agencity

# --- metrics ---
from .metrics import (
    agencity_mean,
    agencity_variance,
    agencity_peak,
    agencity_energy,
    global_efficiency,
    agencity_components,
)

# --- analysis ---
from .regimes import classify_regime
from .stability import stability_summary
from .diagnostics import summarize_diagnostics

# --- information ---
from .information import (
    agencity_information_index,
    agencity_information_density,
)
from agencitylab.analysis.information.agencity_info import full_information_summary
# --- advanced ---
from .geometry import geometric_summary
from .coherence import full_coherence
from .complexity import complexity_summary
from .correlation import full_correlation_summary
from .anomalies import anomaly_summary

# --- optional ---
from .signature import agencity_signature


def build_report_dict(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Build a complete structured scientific report.
    """

    b = result.b
    P_c = getattr(result, "P_c", None)

    bx, by, mag, phase = agencity_components(b)

    beta = getattr(result, "beta", None)
    J = getattr(result, "J", None)

    summary = result.summary() if hasattr(result, "summary") else {}
    metadata = (
        result.metadata.to_dict()
        if hasattr(result, "metadata") and hasattr(result.metadata, "to_dict")
        else {}
    )

    report = {
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
        },

        "regime": classify_regime(b),
        "stability": stability_summary(b),

        "information": {
            "entropy": agencity_information_index(b),
            "density": agencity_information_density(b),
            **full_information_summary(b),
        },

        "geometry": geometric_summary(b),
        "coherence": full_coherence(b),
        "complexity": complexity_summary(b),
        "correlation": full_correlation_summary(b, beta=beta, J=J),
        "anomalies": anomaly_summary(b),

        "diagnostics": summarize_diagnostics(b),
    }

    # optional enrichments
    if J is not None:
        report["metrics"]["J_mean"] = float(np.mean(J))

    if beta is not None:
        report["metrics"]["beta_mean"] = float(np.mean(np.abs(beta)))

    if signature is not None:
        report["signature"] = signature

    if multiscale is not None:
        report["multiscale"] = multiscale

    return report

# ============================================================
# TEXT REPORT
# ============================================================

def build_text_report(
    result,
    *,
    signature: Optional[dict] = None,
    multiscale: Optional[dict] = None,
) -> str:
    """
    Build a human-readable report from the structured report.
    """

    report = build_report_dict(
        result,
        signature=signature,
        multiscale=multiscale,
    )

    s = report.get("summary", {})
    m = report.get("metrics", {})
    st = report.get("stability", {})
    info = report.get("information", {})
    diag = report.get("diagnostics", {})

    lines = [
        "AgencityLab Scientific Report",
        "=" * 45,

        "\n--- Summary ---",
        f"Samples           : {s.get('n_samples', 0)}",
        f"Tau               : {s.get('tau', 0.0):.6f}",

        "\n--- Metrics ---",
        f"Mean |b|          : {m.get('mean_magnitude', 0):.6f}",
        f"Variance |b|      : {m.get('variance_magnitude', 0):.6f}",
        f"Peak              : {m.get('peak', 0):.6f}",
        f"Energy            : {m.get('energy', 0):.6f}",
        f"Efficiency        : {m.get('efficiency', 0)}",

        "\n--- Regime ---",
        f"{report.get('regime', 'unknown')}",

        "\n--- Stability ---",
        f"Trend             : {st.get('trend', 0):.6g}",
        f"Variance          : {st.get('variance', 0):.6g}",

        "\n--- Information ---",
        f"Entropy           : {info.get('entropy', 0):.6f}",
        f"Density           : {info.get('density', 0):.6f}",

        "\n--- Diagnostics ---",
        f"Events            : {diag.get('event_count', 0)}",
        f"Transitions       : {diag.get('transition_count', 0)}",
    ]

    # Optional enrichments
    if "signature" in report:
        sig = report["signature"]
        lines.append("\n--- Signature ---")
        lines.append(f"Slope α           : {sig.get('slope', 0):.8f}")

    if "multiscale" in report:
        ms = report["multiscale"]
        lines.append("\n--- Multiscale ---")
        lines.append(f"tau_opt           : {ms.get('tau_opt', 0)}")

    return "\n".join(lines)