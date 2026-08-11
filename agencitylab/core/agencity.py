"""Observable agencity flux and legacy compatibility helpers.

The canonical scalar operator defined here is ``b = P_c * beta``. The sole
reference canonical end-to-end ``u -> b`` orchestration is
:func:`agencitylab.api.compute.compute_agencity`. Historical full-pipeline and
real-agencity helpers remain only for compatibility and are explicitly
non-canonical diagnostics/wrappers.
"""

from __future__ import annotations

import warnings

import numpy as np

from .coherence import circular_variance, compute_theta, directional_stability, phase_coherence
from .validation import validate_axis, validate_nonnegative_scalar, validate_positive_scalar


def _validate_power_input(P_c, *, expected_shape):
    """Validate finite scalar or sampled ``P_c >= 0`` without altering values."""
    try:
        power = np.asarray(P_c, dtype=float)
    except Exception as exc:
        raise ValueError("P_c must be numeric") from exc

    if power.ndim == 0:
        return validate_nonnegative_scalar(float(power), name="P_c")
    if power.ndim != 1 or power.shape != expected_shape:
        raise ValueError("time-varying P_c must have the same one-dimensional shape as beta")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("P_c must contain only non-negative finite values")
    return power


def agencity(beta_signal, P_c=1.0, *, smooth=False, resolution_scale=None, verbose=False):
    """Compute the canonical observable flux ``b(t) = P_c(t) * beta(t)`` exactly.

    ``P_c`` may be a finite non-negative scalar or sampled profile with the same
    shape as ``beta_signal``. In particular ``P_c = 0`` gives ``b = 0`` exactly.
    No signal-derived power and no epsilon replacement are introduced here.
    """
    if smooth or resolution_scale is not None:
        raise ValueError("canonical b = P_c * beta cannot be smoothed")
    beta_signal = np.asarray(beta_signal, dtype=complex)
    if beta_signal.ndim != 1 or beta_signal.size == 0 or not np.all(np.isfinite(beta_signal)):
        raise ValueError("beta_signal must be a non-empty finite one-dimensional array")
    power = _validate_power_input(P_c, expected_shape=beta_signal.shape)
    b = power * beta_signal
    if verbose:
        print(f"[agencity] |b| mean={np.mean(np.abs(b)):.6g}")
    return b


def agencity_rate(beta_signal, *, delta_star=1.0, axis_star=None, fill_value=np.nan, verbose=False):
    """Diagnostic derivative of beta with respect to reduced time."""
    beta_signal = np.asarray(beta_signal).ravel()
    if beta_signal.size < 2:
        raise ValueError("beta_signal must contain at least two samples")
    if axis_star is None:
        step = validate_positive_scalar(delta_star, name="delta_star")
        out = np.gradient(beta_signal, step)
    else:
        axis_star = validate_axis(axis_star, expected_length=len(beta_signal), name="axis_star")
        out = np.gradient(beta_signal, axis_star)
    if fill_value is not np.nan:
        out = np.where(np.isfinite(out), out, fill_value)
    if verbose:
        print(f"[agencity_rate] mean={np.mean(np.abs(out)):.6g}")
    return out


def decompose_agencity(b, *, verbose=False):
    """Return real part, imaginary part, and magnitude of complex agencity."""
    b = np.asarray(b)
    bx, by, mag = np.real(b), np.imag(b), np.abs(b)
    if verbose:
        print(f"[decomposition] |b| mean={np.mean(mag):.6g}")
    return bx, by, mag


def _legacy_agencity_criteria(
    M,
    O,
    S,
    b,
    *,
    s_threshold=0.0,
    theta_variance_threshold=0.5,
    b_threshold=0.0,
    verbose=False,
):
    M = np.asarray(M, dtype=float)
    O = np.asarray(O, dtype=float)
    S = np.asarray(S, dtype=float)
    b = np.asarray(b, dtype=complex)
    theta = compute_theta(M, O)
    sigma_theta = float(circular_variance(theta))
    theta_coherence = float(phase_coherence(theta))
    theta_stability = float(directional_stability(theta))
    mean_S = float(np.mean(S))
    mean_b = float(np.mean(np.abs(b)))
    has_structure = mean_S > float(s_threshold)
    stable_orientation = sigma_theta < float(theta_variance_threshold)
    significant_flow = mean_b > float(b_threshold)
    score = int(has_structure) + int(stable_orientation) + int(significant_flow)
    out = {
        "real_agencity": bool(has_structure and stable_orientation and significant_flow),
        "has_structure": bool(has_structure),
        "stable_orientation": bool(stable_orientation),
        "significant_flow": bool(significant_flow),
        "score": score,
        "score_normalized": float(score / 3.0),
        "mean_S": mean_S,
        "sigma_theta": sigma_theta,
        "theta_coherence": theta_coherence,
        "theta_stability": theta_stability,
        "mean_abs_b": mean_b,
        "status": "legacy diagnostic; thresholds are not canonical constants",
    }
    if verbose:
        print(f"[legacy agencity criteria] real_agencity={out['real_agencity']}")
    return out


def agencity_criteria(
    M,
    O,
    S,
    b,
    *,
    s_threshold=0.0,
    theta_variance_threshold=0.5,
    b_threshold=0.0,
    verbose=False,
):
    """Legacy real-agencity diagnostic retained for compatibility.

    This helper is not canonical physics. New code must use the modern
    ``agencitylab.analysis`` diagnostics, whose contextual thresholds are
    explicit and which do not invent universal real-agencity constants.
    """
    warnings.warn(
        "agencitylab.core.agencity_criteria is a legacy diagnostic; use "
        "agencitylab.analysis / analyze_coherence instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _legacy_agencity_criteria(
        M,
        O,
        S,
        b,
        s_threshold=s_threshold,
        theta_variance_threshold=theta_variance_threshold,
        b_threshold=b_threshold,
        verbose=verbose,
    )


def compute_full_agencity(
    t,
    u,
    *,
    tau="auto",
    P_c="auto",
    A_ref="auto",
    w=None,
    activity_factor="auto",
    domain=None,
    mechanism=None,
    system=None,
    resolution_scale=None,
    smooth=False,
    verbose=False,
):
    """Legacy compatibility wrapper around the reference public pipeline.

    This function no longer owns or duplicates an end-to-end physical
    interpretation. It delegates to :func:`agencitylab.api.compute.compute_agencity`,
    which is the sole reference canonical ``u -> b`` pipeline. An explicitly
    supplied ``w`` is preserved even when ``w != tau``; omission uses the public
    implementation fallback ``w = tau``.

    The historical ``criteria`` field is retained only as a legacy diagnostic
    payload. It is not part of the canonical computation and new code should run
    diagnostics through :mod:`agencitylab.analysis`.
    """
    warnings.warn(
        "compute_full_agencity is a legacy compatibility wrapper; use "
        "agencitylab.compute_agencity for the reference canonical pipeline",
        DeprecationWarning,
        stacklevel=2,
    )
    if smooth:
        raise ValueError("legacy compute_full_agencity cannot smooth the canonical pipeline")

    # Lazy import avoids an import cycle while ensuring this historical entry
    # point delegates to, rather than reimplements, the reference orchestration.
    from agencitylab.api.compute import compute_agencity

    result = compute_agencity(
        u=u,
        xi=t,
        tau=tau,
        P_c=P_c,
        A_ref=A_ref,
        w=w,
        activity_factor=activity_factor,
        domain=domain,
        mechanism=mechanism,
        system_type=system,
        resolution_scale=resolution_scale,
        verbose=verbose,
    )

    return {
        "u_star": result.u_star,
        "A_ref": result.A_ref,
        "tau": result.tau,
        "w": result.memory_window,
        "t_star": result.t_star,
        "P_c": result.P_c,
        "X": result.X_star,
        "A": result.A_star,
        "M": result.M,
        "O": result.O,
        "D": result.D,
        "S": result.S,
        "J": result.J,
        "U": result.U,
        "beta": result.beta,
        "b": result.b,
        "criteria": _legacy_agencity_criteria(result.M, result.O, result.S, result.b),
        "canonical_reference": "agencitylab.compute_agencity",
        "status": "legacy compatibility wrapper; not an independent canonical pipeline",
    }
