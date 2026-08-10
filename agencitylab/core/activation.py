"""
activation.py

Activation utilities for AgencityLab.

Canonical theory
----------------
    X*(xi) = d(u*) / d(xi)

IMPORTANT
---------
Modern Agencity theory separates:

    - observable normalization,
    - local dynamics,
    - structural memory.

Therefore:
    tau does NOT directly scale activation.

tau intervenes later through:
    - CRM structural memory,
    - multiscale organization,
    - temporal structural analysis.

Physical interpretation
-----------------------
Activation measures local observable variation.

The optional resolution_scale parameter does NOT represent
arbitrary denoising.

It represents:
    - instrumental bandwidth,
    - observational resolution,
    - physical coarse-graining,
    - finite structural perception scale.

This preserves physical consistency while preventing
non-physical derivative explosions on unresolved noise.
"""

from __future__ import annotations

import numpy as np

from .validation import (
    validate_axis,
    validate_signal,
)

from .safeguards import (
    EPS,
    replace_non_finite,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _infer_step(axis):
    """
    Infer characteristic sampling step.
    """

    diffs = np.diff(axis)

    diffs = diffs[
        np.isfinite(diffs)
        & (np.abs(diffs) > EPS)
    ]

    if diffs.size == 0:
        return 1.0

    return float(
        np.median(
            np.abs(diffs)
        )
    )


def _moving_average(
    x,
    window,
):
    """
    Physical coarse-graining operator.

    This represents finite observational resolution.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    window = int(window)

    if window <= 1:
        return x.copy()

    kernel = (
        np.ones(window)
        / float(window)
    )

    return np.convolve(
        x,
        kernel,
        mode="same",
    )


def _resolve_signal(
    signal,
    axis,
    resolution_scale=None,
):
    """
    Apply physical observational resolution.

    Parameters
    ----------
    resolution_scale :
        Physical coarse-graining scale.
    """

    x = np.asarray(
        signal,
        dtype=float,
    )

    if resolution_scale is None:
        return x

    resolution_scale = float(
        resolution_scale
    )

    if resolution_scale <= 0:

        raise ValueError(
            "resolution_scale must be positive"
        )

    # ========================================================
    # STEP
    # ========================================================

    if np.ndim(axis) == 0:

        step = float(axis)

    else:

        step = _infer_step(axis)

    # ========================================================
    # WINDOW
    # ========================================================

    n = max(
        1,
        int(
            round(
                resolution_scale
                / max(step, EPS)
            )
        ),
    )

    return _moving_average(
        x,
        n,
    )


def _safe_gradient(
    signal,
    axis,
):
    """
    Numerically stable derivative.
    """

    if np.ndim(axis) == 0:

        step = float(axis)

        return np.gradient(
            signal,
            step,
            edge_order=2
            if signal.size > 2
            else 1,
        )

    axis = validate_axis(
        axis,
        expected_length=signal.size,
        name="axis",
    )

    return np.gradient(
        signal,
        axis,
        edge_order=2
        if signal.size > 2
        else 1,
    )


# ============================================================
# REDUCED COORDINATE
# ============================================================

def reduced_coordinate(
    axis,
    tau,
):
    """
    Compute reduced coordinate:

        xi* = xi / tau

    This remains useful for:
        - structural visualization,
        - reduced representations,
        - multiscale plotting.

    IMPORTANT
    ---------
    Modern activation dynamics are computed with respect
    to the physical observation axis xi,
    not xi*.
    """

    axis = np.asarray(
        axis,
        dtype=float,
    )

    tau = float(tau)

    if tau <= 0:

        raise ValueError(
            "tau must be positive"
        )

    return axis / tau


# ============================================================
# ACTIVATION
# ============================================================

def compute_activation(
    u_star,
    axis,
    *,
    resolution_scale=None,
    replace_nan=True,
    clip=None,
    verbose=False,
):
    """
    Compute activation field.

    Canonical definition
    --------------------
        X* = d(u*) / d(xi)

    Parameters
    ----------
    u_star :
        Normalized observable.

    axis :
        Observation coordinate or constant step.

    resolution_scale :
        Physical observation scale.

    replace_nan :
        Replace invalid values.

    clip :
        Optional clipping magnitude.

    Returns
    -------
    X : ndarray
        Activation field.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    u_star = validate_signal(
        u_star,
        name="u_star",
    ).ravel()

    u_star = np.asarray(
        u_star,
        dtype=float,
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    if replace_nan:

        u_star = replace_non_finite(
            u_star,
            default=0.0,
        )

    # ========================================================
    # PHYSICAL RESOLUTION
    # ========================================================

    u_resolved = _resolve_signal(
        u_star,
        axis,
        resolution_scale=resolution_scale,
    )

    # ========================================================
    # DERIVATIVE
    # ========================================================

    X = _safe_gradient(
        u_resolved,
        axis,
    )

    # ========================================================
    # OPTIONAL CLIPPING
    # ========================================================

    if clip is not None:

        clip = abs(float(clip))

        X = np.clip(
            X,
            -clip,
            clip,
        )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    X = replace_non_finite(
        X,
        default=0.0,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            "[activation] "
            f"mean={np.nanmean(X):.6f}"
        )

        print(
            "[activation] "
            f"std={np.nanstd(X):.6f}"
        )

        print(
            "[activation] "
            f"min={np.nanmin(X):.6f}"
        )

        print(
            "[activation] "
            f"max={np.nanmax(X):.6f}"
        )

        if resolution_scale is not None:

            print(
                "[activation] "
                f"resolution_scale="
                f"{resolution_scale}"
            )

    return X


# ============================================================
# PUBLIC API
# ============================================================

def activation(
    u_star,
    axis,
    *,
    resolution_scale=None,
    replace_nan=True,
    clip=None,
    verbose=False,
):
    """
    Canonical public API.
    """

    return compute_activation(
        u_star,
        axis,
        resolution_scale=resolution_scale,
        replace_nan=replace_nan,
        clip=clip,
        verbose=verbose,
    )


def activation_from_signal(
    u_star,
    axis,
    *,
    resolution_scale=None,
    replace_nan=True,
    clip=None,
    verbose=False,
):
    """
    Alias for pipeline usage.
    """

    return compute_activation(
        u_star,
        axis,
        resolution_scale=resolution_scale,
        replace_nan=replace_nan,
        clip=clip,
        verbose=verbose,
    )