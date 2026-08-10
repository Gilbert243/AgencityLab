"""
contrast.py

Contrast operator for AgencityLab.

Canonical theory
----------------
    J = ln((e + D) / (e + S))

where:
    D = dynamic intensity
    S = structural intensity

Physical interpretation
-----------------------
J measures the balance between:

    - local dynamical activity,
    - structural coherence.

Interpretation
--------------
    J > 0 :
        dynamics dominate structure

    J < 0 :
        structure dominates dynamics

    J ≈ 0 :
        dynamic/structural equilibrium

Important
---------
The exponential offset:

    e

is part of the canonical theory.

It prevents:
    - singularities,
    - undefined logarithms,
    - excessive amplification near zero.

This implementation additionally supports:
    - robust saturation,
    - clipping,
    - physical diagnostics.

without modifying canonical behavior.
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
    D,
    S,
):
    """
    Validate compatible intensity fields.
    """

    D = np.asarray(
        D,
        dtype=float,
    )

    S = np.asarray(
        S,
        dtype=float,
    )

    if D.shape != S.shape:

        raise ValueError(
            "D and S must share the same shape"
        )

    return D, S


# ============================================================
# CONTRAST
# ============================================================

def compute_contrast(
    D,
    S,
    *,
    eps: float = EPS,
    robust=False,
    robust_scale=10.0,
    clip=None,
    return_ratio=False,
    verbose=False,
):
    """
    Compute logarithmic structural contrast.

    Canonical definition
    --------------------
        J = ln((e + D) / (e + S))

    Parameters
    ----------
    D :
        Dynamic intensity field.

    S :
        Structural intensity field.

    eps :
        Numerical safeguard.

    robust :
        Apply bounded saturation.

    robust_scale :
        Saturation scale.

    clip :
        Optional clipping magnitude.

    return_ratio :
        Return raw contrast ratio.

    Returns
    -------
    J : ndarray

    optionally:
        (J, ratio)
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    D, S = _validate_pair(
        D,
        S,
    )

    eps = max(
        float(eps),
        EPS,
    )

    # ========================================================
    # CANONICAL RATIO
    # ========================================================

    ratio = (
        np.e + D + eps
    ) / (
        np.e + S + eps
    )

    # ========================================================
    # CONTRAST
    # ========================================================

    J = np.log(ratio)

    # ========================================================
    # OPTIONAL ROBUST SATURATION
    # ========================================================

    if robust:

        scale = max(
            float(robust_scale),
            EPS,
        )

        J = scale * np.tanh(
            J / scale
        )

    # ========================================================
    # OPTIONAL CLIPPING
    # ========================================================

    if clip is not None:

        clip = abs(float(clip))

        J = np.clip(
            J,
            -clip,
            clip,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    J = replace_non_finite(
        J,
        0.0,
    )

    J = np.asarray(
        J,
        dtype=float,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        positive_fraction = np.mean(
            J > 0
        )

        negative_fraction = np.mean(
            J < 0
        )

        equilibrium_fraction = np.mean(
            np.abs(J) < 1e-3
        )

        print(
            "[contrast] "
            "Computing contrast J"
        )

        print(
            "[contrast] "
            f"mean={np.mean(J):.6f}"
        )

        print(
            "[contrast] "
            f"std={np.std(J):.6f}"
        )

        print(
            "[contrast] "
            f"min={np.min(J):.6f}"
        )

        print(
            "[contrast] "
            f"max={np.max(J):.6f}"
        )

        print(
            "[contrast] "
            f"positive_fraction="
            f"{positive_fraction:.6f}"
        )

        print(
            "[contrast] "
            f"negative_fraction="
            f"{negative_fraction:.6f}"
        )

        print(
            "[contrast] "
            f"equilibrium_fraction="
            f"{equilibrium_fraction:.6f}"
        )

        if robust:

            print(
                "[contrast] "
                f"robust_scale="
                f"{robust_scale}"
            )

    # ========================================================
    # RETURN
    # ========================================================

    if return_ratio:
        return J, ratio

    return J