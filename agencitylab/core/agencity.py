"""Observable agencity flux and canonical end-to-end core pipeline."""

from __future__ import annotations

import numpy as np

from .coherence import circular_variance, compute_theta, directional_stability, phase_coherence
from .power import characteristic_power
from .tau import characteristic_time
from .validation import is_exactly_constant, validate_axis, validate_positive_scalar


def agencity(beta_signal, P_c=1.0, *, smooth=False, resolution_scale=None, verbose=False):
    """Compute the canonical observable flux ``b = P_c * beta`` exactly."""
    if smooth or resolution_scale is not None:
        raise ValueError("canonical b = P_c * beta cannot be smoothed")
    beta_signal = np.asarray(beta_signal, dtype=complex)
    if beta_signal.size == 0 or not np.all(np.isfinite(beta_signal)):
        raise ValueError("beta_signal must be non-empty and finite")
    power = validate_positive_scalar(P_c, name="P_c")
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
    """Diagnostic real-agencity criterion; thresholds are analysis choices, not core physics."""
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
    }
    if verbose:
        print(f"[agencity criteria] real_agencity={out['real_agencity']}")
    return out


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
    """Run the canonical ``u -> b`` construction and return all intermediate fields.

    An exactly constant sampled observable is treated as the canonical null-state
    postulate. The derivative and CRM stages are bypassed and all dynamical,
    structural, and agencity fields are set exactly to zero.
    """
    del mechanism
    if activity_factor not in {None, "auto"}:
        raise ValueError("activity_factor is not part of the canonical CRM")
    if resolution_scale is not None or smooth:
        raise ValueError("preprocessing/smoothing is outside the canonical core")

    from .activation import activation, reduced_coordinate
    from .activity import activity
    from .beta import compute_beta
    from .intensity import compute_intensities
    from .memory import memory
    from .normalization import normalize_signal
    from .organization import organization

    t = validate_axis(t, name="t")
    u_star, A_ref_resolved = normalize_signal(
        u, A_ref=A_ref, domain=domain, method="canonical", verbose=verbose
    )
    tau_value = characteristic_time(tau=tau, system=system, domain=domain, verbose=verbose)
    if w is not None and float(w) != float(tau_value):
        raise ValueError("canonical memory window is fixed by w = tau")
    memory_window = tau_value
    power = characteristic_power(
        value=None if P_c in {None, "auto"} else P_c,
        system=system,
        domain=domain,
        tau=tau_value,
        verbose=verbose,
    )
    t_star = reduced_coordinate(t, tau_value)

    if is_exactly_constant(u_star):
        if verbose:
            print("[canonical] exact rest state detected; derivative/CRM stages bypassed")
        zeros = np.zeros_like(u_star, dtype=float)
        complex_zeros = np.zeros_like(u_star, dtype=complex)
        X_star = zeros.copy()
        A_star = zeros.copy()
        M = zeros.copy()
        O = zeros.copy()
        D = zeros.copy()
        S = zeros.copy()
        J = zeros.copy()
        U = complex_zeros.copy()
        beta_signal = complex_zeros.copy()
        b = complex_zeros.copy()
    else:
        X_star = activation(u_star, axis=t_star, verbose=verbose)
        A_star = activity(X_star, axis=t_star, verbose=verbose)
        M = memory(u_star, tau_value, axis=t, window=memory_window, verbose=verbose)
        O = organization(u_star, X_star, tau_value, axis=t, window=memory_window, verbose=verbose)
        D, S = compute_intensities(X_star, A_star, M, O, verbose=verbose)
        J, U, beta_signal = compute_beta(D, S, M, O, verbose=verbose)
        b = agencity(beta_signal, power, verbose=verbose)

    return {
        "u_star": u_star,
        "A_ref": A_ref_resolved,
        "tau": tau_value,
        "w": memory_window,
        "t_star": t_star,
        "P_c": power,
        "X": X_star,
        "A": A_star,
        "M": M,
        "O": O,
        "D": D,
        "S": S,
        "J": J,
        "U": U,
        "beta": beta_signal,
        "b": b,
        "criteria": agencity_criteria(M, O, S, b, verbose=verbose),
    }
