"""
orientation.py

Orientation of Agencity.

Canonical theory
----------------
    U = (M + i O) / S

where:
    S = sqrt(M² + O²)

U is a complex orientation field.

Important
---------
U exists ONLY when structural intensity exists.

If:
    S = 0

then:
    U = 0

This preserves the canonical interpretation:

    no memory
    +
    no organization
    =>
    no orientation

This implementation intentionally avoids
injecting numerical epsilons inside the
physical canonical equations.

EPS remains only:
    - a numerical safeguard,
    - never a physical quantity.
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

def _safe_structural_norm(
    M,
    O,
):
    """
    Structural intensity:

        S = sqrt(M² + O²)
    """

    return np.sqrt(
        M**2 + O**2
    )


# ============================================================
# ORIENTATION
# ============================================================

def compute_orientation(
    M,
    O,
    *,
    null_threshold=None,
    return_intensity=False,
    return_mask=False,
    verbose=False,
):
    """
    Compute complex orientation field U.

    Canonical definition
    --------------------
        U = (M + i O) / S

    where:
        S = sqrt(M² + O²)

    Canonical null regime
    ---------------------
    If:
        S <= threshold

    then:
        U = 0

    Parameters
    ----------
    M : ndarray
        Memory component.

    O : ndarray
        Organization component.

    null_threshold : float or None
        Structural null threshold.

        If None:
            threshold = EPS

    return_intensity : bool
        Return S together with U.

    return_mask : bool
        Return valid structural mask.

    verbose : bool
        Enable diagnostics.

    Returns
    -------
    U : ndarray

    optionally:
        (U, S)

        (U, S, valid)

        (U, valid)
    """

    # ========================================================
    # INPUTS
    # ========================================================

    M = np.asarray(
        M,
        dtype=float,
    )

    O = np.asarray(
        O,
        dtype=float,
    )

    if M.shape != O.shape:

        raise ValueError(
            "M and O must share the same shape"
        )

    threshold = (
        EPS
        if null_threshold is None
        else abs(float(null_threshold))
    )

    # ========================================================
    # STRUCTURAL INTENSITY
    # ========================================================

    S = _safe_structural_norm(
        M,
        O,
    )

    # ========================================================
    # VALID STRUCTURE
    # ========================================================

    valid = S > threshold

    # ========================================================
    # OUTPUT
    # ========================================================

    U = np.zeros_like(
        S,
        dtype=complex,
    )

    # ========================================================
    # CANONICAL ORIENTATION
    # ========================================================

    if np.any(valid):

        U[valid] = (
            M[valid]
            + 1j * O[valid]
        ) / S[valid]

    # ========================================================
    # CLEANUP
    # ========================================================

    U = replace_non_finite(
        U,
        0.0,
    )

    S = replace_non_finite(
        S,
        0.0,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        magnitude = np.abs(U)

        print(
            "[orientation] "
            "Computing orientation U"
        )

        print(
            "[orientation] "
            f"valid_fraction="
            f"{np.mean(valid):.6f}"
        )

        print(
            "[orientation] "
            f"|U| mean="
            f"{np.mean(magnitude):.6f}"
        )

        print(
            "[orientation] "
            f"|U| std="
            f"{np.std(magnitude):.6f}"
        )

        print(
            "[orientation] "
            f"|U| min="
            f"{np.min(magnitude):.6f}"
        )

        print(
            "[orientation] "
            f"|U| max="
            f"{np.max(magnitude):.6f}"
        )

        print(
            "[orientation] "
            f"S mean="
            f"{np.mean(S):.6f}"
        )

        print(
            "[orientation] "
            f"S std="
            f"{np.std(S):.6f}"
        )

        print(
            "[orientation] "
            f"S min="
            f"{np.min(S):.6f}"
        )

        print(
            "[orientation] "
            f"S max="
            f"{np.max(S):.6f}"
        )

        print(
            "[orientation] "
            f"threshold={threshold:.6e}"
        )

    # ========================================================
    # RETURNS
    # ========================================================

    if return_intensity and return_mask:
        return U, S, valid

    if return_intensity:
        return U, S

    if return_mask:
        return U, valid

    return U


# ============================================================
# ANGLE
# ============================================================

def compute_angle(
    b,
    *,
    degrees=False,
    unwrap=False,
    verbose=False,
):
    """
    Compute orientation angle theta.

    Parameters
    ----------
    b : ndarray
        Complex agencity field.

    degrees : bool
        Return degrees instead of radians.

    unwrap : bool
        Remove 2π discontinuities.

    Returns
    -------
    theta : ndarray
    """

    b = np.asarray(
        b
    )

    theta = np.angle(b)

    # ========================================================
    # OPTIONAL UNWRAP
    # ========================================================

    if unwrap:
        theta = np.unwrap(theta)

    # ========================================================
    # OPTIONAL DEGREES
    # ========================================================

    if degrees:
        theta = np.degrees(theta)

    # ========================================================
    # CLEANUP
    # ========================================================

    theta = replace_non_finite(
        theta,
        0.0,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        unit = (
            "deg"
            if degrees
            else "rad"
        )

        print(
            "[angle] "
            f"theta mean="
            f"{np.mean(theta):.6f} "
            f"({unit})"
        )

        print(
            "[angle] "
            f"theta std="
            f"{np.std(theta):.6f} "
            f"({unit})"
        )

        print(
            "[angle] "
            f"theta min="
            f"{np.min(theta):.6f} "
            f"({unit})"
        )

        print(
            "[angle] "
            f"theta max="
            f"{np.max(theta):.6f} "
            f"({unit})"
        )

    return theta