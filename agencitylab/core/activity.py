"""
activity.py

Activity utilities for AgencityLab.

Canonical theory
----------------
    A*(xi) = d(X*) / d(xi)

IMPORTANT
---------
Modern Agencity theory separates:

    - observable normalization,
    - local activation dynamics,
    - structural memory.

Therefore:
    tau does NOT directly scale activity.

tau intervenes later through:
    - CRM structural memory,
    - multiscale organization,
    - temporal structural analysis.

Physical interpretation
-----------------------
Activity measures local variation of activation.

Because activity is a second-order derivative quantity,
it is highly sensitive to unresolved noise.

The optional resolution_scale parameter therefore represents:

    - instrumental bandwidth,
    - observational resolution,
    - finite physical perception scale,
    - structural coarse-graining.

This is NOT arbitrary denoising.

It is a physical-resolution operator.
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

from .activation import (
    _resolve_signal,
    _safe_gradient,
)


# ============================================================
# ACTIVITY
# ============================================================

def compute_activity(
    X_star,
    axis,
    *,
    resolution_scale=None,
    replace_nan=True,
    clip=None,
    verbose=False,
):
    """
    Compute activity field.

    Canonical definition
    --------------------
        A* = d(X*) / d(xi)

    Parameters
    ----------
    X_star :
        Activation field.

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
    A : ndarray
        Activity field.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    X_star = validate_signal(
        X_star,
        name="X_star",
    ).ravel()

    X_star = np.asarray(
        X_star,
        dtype=float,
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    if replace_nan:

        X_star = replace_non_finite(
            X_star,
            default=0.0,
        )

    # ========================================================
    # PHYSICAL RESOLUTION
    # ========================================================

    X_resolved = _resolve_signal(
        X_star,
        axis,
        resolution_scale=resolution_scale,
    )

    # ========================================================
    # DERIVATIVE
    # ========================================================

    A = _safe_gradient(
        X_resolved,
        axis,
    )

    # ========================================================
    # OPTIONAL CLIPPING
    # ========================================================

    if clip is not None:

        clip = abs(float(clip))

        A = np.clip(
            A,
            -clip,
            clip,
        )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    A = replace_non_finite(
        A,
        default=0.0,
    )

    A = np.asarray(
        A,
        dtype=float,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            "[activity] "
            f"mean={np.nanmean(A):.6f}"
        )

        print(
            "[activity] "
            f"std={np.nanstd(A):.6f}"
        )

        print(
            "[activity] "
            f"min={np.nanmin(A):.6f}"
        )

        print(
            "[activity] "
            f"max={np.nanmax(A):.6f}"
        )

        if resolution_scale is not None:

            print(
                "[activity] "
                f"resolution_scale="
                f"{resolution_scale}"
            )

    return A


# ============================================================
# PUBLIC API
# ============================================================

def activity(
    X_star,
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

    return compute_activity(
        X_star,
        axis,
        resolution_scale=resolution_scale,
        replace_nan=replace_nan,
        clip=clip,
        verbose=verbose,
    )


def activity_from_signal(
    X_star,
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

    return compute_activity(
        X_star,
        axis,
        resolution_scale=resolution_scale,
        replace_nan=replace_nan,
        clip=clip,
        verbose=verbose,
    )