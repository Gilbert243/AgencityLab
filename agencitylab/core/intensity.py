"""
intensity.py

Intensity measures for AgencityLab.

Canonical theory
----------------
Dynamic intensity:

    D = sqrt(X² + (A X)²)

Structural intensity:

    S = sqrt(M² + O²)

where:
    D = dynamic intensity
    S = structural intensity

Physical interpretation
-----------------------
D measures local dynamical activity.

S measures structural coherence.

Important
---------
Dynamic intensity may become numerically unstable
when unresolved noise generates extremely large
second-order derivatives.

This implementation therefore supports:

    - physical clipping,
    - finite-resolution regularization,
    - optional robust saturation.

These operations represent:
    - instrumental limits,
    - finite observational resolution,
    - physical coarse-graining.

They are NOT arbitrary denoising.
"""

from __future__ import annotations

import numpy as np

from .safeguards import (
    EPS,
    replace_non_finite,
)


# ============================================================
# INTERNAL
# ============================================================

def _validate_pair(
    a,
    b,
    name_a,
    name_b,
):
    """
    Validate compatible arrays.
    """

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if a.shape != b.shape:

        raise ValueError(
            f"{name_a} and {name_b} "
            f"must share the same shape"
        )

    return a, b


def _safe_norm2(
    a,
    b,
):
    """
    Stable Euclidean norm.
    """

    return np.sqrt(
        a * a
        + b * b
    )


# ============================================================
# DYNAMIC INTENSITY
# ============================================================

def compute_dynamic_intensity(
    X,
    A,
    *,
    clip=None,
    robust=False,
    robust_scale=10.0,
    return_components=False,
    verbose=False,
):
    """
    Compute dynamic intensity.

    Canonical definition
    --------------------
        D = sqrt(X² + (A X)²)

    Parameters
    ----------
    X :
        Activation field.

    A :
        Activity field.

    clip :
        Optional clipping magnitude.

    robust :
        Apply bounded physical saturation.

    robust_scale :
        Saturation scale.

    return_components :
        Return AX contribution.

    Returns
    -------
    D : ndarray

    optionally:
        (D, AX)
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    X, A = _validate_pair(
        X,
        A,
        "X",
        "A",
    )

    # ========================================================
    # DYNAMIC COUPLING
    # ========================================================

    AX = A * X

    # ========================================================
    # OPTIONAL ROBUST SATURATION
    # ========================================================

    if robust:

        scale = max(
            float(robust_scale),
            EPS,
        )

        AX = scale * np.tanh(
            AX / scale
        )

    # ========================================================
    # INTENSITY
    # ========================================================

    D = _safe_norm2(
        X,
        AX,
    )

    # ========================================================
    # OPTIONAL CLIPPING
    # ========================================================

    if clip is not None:

        clip = abs(float(clip))

        D = np.clip(
            D,
            0.0,
            clip,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    D = replace_non_finite(
        D,
        0.0,
    )

    D = np.asarray(
        D,
        dtype=float,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            "[intensity] "
            "Computing dynamic intensity D"
        )

        print(
            "[intensity] "
            f"D mean={np.mean(D):.6f}"
        )

        print(
            "[intensity] "
            f"D std={np.std(D):.6f}"
        )

        print(
            "[intensity] "
            f"D min={np.min(D):.6f}"
        )

        print(
            "[intensity] "
            f"D max={np.max(D):.6f}"
        )

        print(
            "[intensity] "
            f"|AX| mean="
            f"{np.mean(np.abs(AX)):.6f}"
        )

        if robust:

            print(
                "[intensity] "
                f"robust_scale="
                f"{robust_scale}"
            )

    # ========================================================
    # RETURN
    # ========================================================

    if return_components:
        return D, AX

    return D


# ============================================================
# STRUCTURAL INTENSITY
# ============================================================

def compute_structural_intensity(
    M,
    O,
    *,
    null_threshold=None,
    return_mask=False,
    verbose=False,
):
    """
    Compute structural intensity.

    Canonical definition
    --------------------
        S = sqrt(M² + O²)

    Parameters
    ----------
    M :
        Memory field.

    O :
        Organization field.

    null_threshold :
        Structural null threshold.

    return_mask :
        Return valid structure mask.

    Returns
    -------
    S : ndarray

    optionally:
        (S, mask)
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    M, O = _validate_pair(
        M,
        O,
        "M",
        "O",
    )

    threshold = (
        EPS
        if null_threshold is None
        else float(null_threshold)
    )

    # ========================================================
    # STRUCTURAL NORM
    # ========================================================

    S = _safe_norm2(
        M,
        O,
    )

    mask = S > threshold

    # ========================================================
    # CLEANUP
    # ========================================================

    S = replace_non_finite(
        S,
        0.0,
    )

    S = np.asarray(
        S,
        dtype=float,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            "[intensity] "
            "Computing structural intensity S"
        )

        print(
            "[intensity] "
            f"S mean={np.mean(S):.6f}"
        )

        print(
            "[intensity] "
            f"S std={np.std(S):.6f}"
        )

        print(
            "[intensity] "
            f"S min={np.min(S):.6f}"
        )

        print(
            "[intensity] "
            f"S max={np.max(S):.6f}"
        )

        print(
            "[intensity] "
            f"valid_fraction="
            f"{np.mean(mask):.6f}"
        )

    # ========================================================
    # RETURN
    # ========================================================

    if return_mask:
        return S, mask

    return S


# ============================================================
# COMBINED API
# ============================================================

def compute_intensities(
    X,
    A,
    M,
    O,
    *,
    dynamic_clip=None,
    robust_dynamic=False,
    robust_scale=10.0,
    structural_threshold=None,
    verbose=False,
):
    """
    Compute both dynamic and structural intensities.

    Returns
    -------
    D :
        Dynamic intensity.

    S :
        Structural intensity.
    """

    D = compute_dynamic_intensity(
        X,
        A,
        clip=dynamic_clip,
        robust=robust_dynamic,
        robust_scale=robust_scale,
        verbose=verbose,
    )

    S = compute_structural_intensity(
        M,
        O,
        null_threshold=structural_threshold,
        verbose=verbose,
    )

    return D, S