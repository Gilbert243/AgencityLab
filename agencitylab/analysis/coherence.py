"""
analysis/coherence.py
=====================

Scientific coherence analysis utilities for AgencityLab.

This module provides higher-level coherence diagnostics
built on top of the mathematical primitives implemented in:

    agencitylab.core.coherence

Unlike core/coherence.py, this module is intended for:

    - scientific interpretation,
    - diagnostics,
    - regime characterization,
    - coherence scoring,
    - reporting.

The functions here operate mainly on:
    - beta fields,
    - agencity flows,
    - orientation dynamics,
    - structural stability.

Canonical references
--------------------
Volume 2:
    - Chapter 2 (orientation structure)
    - Chapter 7 (§7.8 real agencity criterion)

Key quantities
---------------
    Θ   : structural orientation
    ΣΘ  : angular variance
    |b| : agencity flow magnitude
"""

from __future__ import annotations

import numpy as np

from agencitylab.core.coherence import (
    compute_theta,
    circular_mean,
    circular_variance,
    circular_std,
    phase_coherence as core_phase_coherence,
    directional_stability,
    angular_variance,
)

EPS = 1e-12


# ============================================================
# PHASE COHERENCE
# ============================================================

def phase_coherence(
    b,
):
    """
    Measure complex phase coherence.

    Parameters
    ----------
    b : ndarray
        Complex agencity field.

    Returns
    -------
    float
        Phase coherence in [0, 1].

    Interpretation
    --------------
    ~1:
        strong global phase alignment

    ~0:
        random phase structure
    """
    b = np.asarray(
        b,
        dtype=complex,
    )

    if b.size == 0:
        return 0.0

    theta = np.angle(b)

    return float(
        np.abs(
            np.mean(
                np.exp(1j * theta)
            )
        )
    )


# ============================================================
# AMPLITUDE COHERENCE
# ============================================================

def amplitude_coherence(
    b,
):
    """
    Measure amplitude stability.

    Parameters
    ----------
    b : ndarray
        Complex agencity field.

    Returns
    -------
    float
        Inverse amplitude variability.
    """
    mag = np.abs(b)

    return float(
        1.0 / (
            np.std(mag)
            + EPS
        )
    )


# ============================================================
# TEMPORAL COHERENCE
# ============================================================

def temporal_coherence(
    b,
):
    """
    Measure temporal coherence using
    lag-1 autocorrelation.

    Parameters
    ----------
    b : ndarray

    Returns
    -------
    float
    """
    b = np.asarray(
        b,
        dtype=complex,
    )

    mag = np.abs(b)

    if len(mag) < 2:
        return 0.0

    corr = np.corrcoef(
        mag[:-1],
        mag[1:],
    )[0, 1]

    if not np.isfinite(corr):
        return 0.0

    return float(corr)


# ============================================================
# STRUCTURAL ORIENTATION ANALYSIS
# ============================================================

def orientation_statistics(
    M,
    O,
):
    """
    Compute structural orientation statistics.

    Parameters
    ----------
    M : ndarray
        Structural memory.

    O : ndarray
        Structural organization.

    Returns
    -------
    dict
    """
    theta = compute_theta(M, O)

    return {
        "theta_mean":
            float(circular_mean(theta)),

        "theta_variance":
            float(circular_variance(theta)),

        "theta_std":
            float(circular_std(theta)),

        "directional_stability":
            float(
                directional_stability(theta)
            ),

        "phase_coherence":
            float(
                core_phase_coherence(theta)
            ),
    }


# ============================================================
# ANGULAR STABILITY
# ============================================================

def angular_stability(
    M,
    O,
    *,
    window=None,
):
    """
    Compute angular stability diagnostics.

    Parameters
    ----------
    M : ndarray

    O : ndarray

    window : int | None

    Returns
    -------
    dict
    """
    theta = compute_theta(M, O)

    sigma_theta = angular_variance(
        theta,
        window=window,
    )

    sigma_mean = np.nanmean(
        sigma_theta
    )

    stability = 1.0 / (
        sigma_mean + EPS
    )

    return {
        "sigma_theta":
            sigma_theta,

        "sigma_theta_mean":
            float(sigma_mean),

        "angular_stability":
            float(stability),
    }


# ============================================================
# REAL AGENCITY CRITERION
# ============================================================

def real_agencity_criterion(
    S,
    theta_variance,
    b,
    *,
    s_threshold=0.0,
    theta_variance_threshold=0.5,
    b_threshold=0.0,
):
    """
    Evaluate the real-agencity criterion.

    Canonical criterion
    -------------------
        Real agencity iff:

            S > 0
            ΣΘ small
            |b| significant

    Parameters
    ----------
    S : ndarray
        Structural intensity.

    theta_variance : float
        Angular variance ΣΘ.

    b : ndarray
        Complex agencity flow.

    Returns
    -------
    dict
    """
    S = np.asarray(
        S,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=complex,
    )

    mean_S = float(
        np.mean(S)
    )

    mean_b = float(
        np.mean(np.abs(b))
    )

    has_structure = (
        mean_S > s_threshold
    )

    stable_orientation = (
        theta_variance
        < theta_variance_threshold
    )

    significant_flow = (
        mean_b > b_threshold
    )

    is_real_agent = (
        has_structure
        and stable_orientation
        and significant_flow
    )

    return {
        "real_agencity":
            bool(is_real_agent),

        "has_structure":
            bool(has_structure),

        "stable_orientation":
            bool(stable_orientation),

        "significant_flow":
            bool(significant_flow),

        "mean_S":
            mean_S,

        "theta_variance":
            float(theta_variance),

        "mean_abs_b":
            mean_b,
    }


# ============================================================
# GLOBAL COHERENCE
# ============================================================

def full_coherence(
    b,
):
    """
    Compute full coherence diagnostics.

    Parameters
    ----------
    b : ndarray

    Returns
    -------
    dict
    """
    return {

        "phase_coherence":
            phase_coherence(b),

        "amplitude_coherence":
            amplitude_coherence(b),

        "temporal_coherence":
            temporal_coherence(b),
    }


# ============================================================
# FULL STRUCTURAL COHERENCE
# ============================================================

def full_structural_coherence(
    M,
    O,
    b,
    *,
    window=None,
):
    """
    Complete structural coherence analysis.

    Parameters
    ----------
    M : ndarray

    O : ndarray

    b : ndarray

    window : int | None

    Returns
    -------
    dict
    """
    orientation = orientation_statistics(
        M,
        O,
    )

    angular = angular_stability(
        M,
        O,
        window=window,
    )

    coherence = full_coherence(
        b,
    )

    return {
        "orientation":
            orientation,

        "angular":
            angular,

        "coherence":
            coherence,
    }