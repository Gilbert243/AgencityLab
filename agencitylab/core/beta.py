"""
beta.py

Structured agencity construction for AgencityLab.

Canonical theory
----------------
    J = contrast(D, S)

    U = orientation(M, O)

    beta = J * U

where:
    J : scalar structural contrast
    U : complex organizational orientation

Physical interpretation
-----------------------
beta represents structured agencity.

It combines:

    - structural imbalance (J),
    - organizational direction (U).

Interpretation
--------------
Large |beta| implies:
    strong organized dynamical asymmetry.

beta ≈ 0 may correspond to:
    - noise,
    - equilibrium,
    - absence of structure,
    - absence of orientation.

Important
---------
This implementation preserves the canonical theory while adding:

    - structural null protection,
    - robust diagnostics,
    - optional saturation,
    - optional clipping,
    - detailed statistics.

No canonical equation is modified.
"""

from __future__ import annotations

import numpy as np

from .contrast import (
    compute_contrast,
)

from .orientation import (
    compute_orientation,
)

from .safeguards import (
    EPS,
    replace_non_finite,
)


# ============================================================
# INTERNAL
# ============================================================

def _safe_beta_stats(beta_signal):
    """
    Compute robust beta diagnostics.
    """

    magnitude = np.abs(beta_signal)

    return {
        "mean": float(np.mean(magnitude)),
        "std": float(np.std(magnitude)),
        "min": float(np.min(magnitude)),
        "max": float(np.max(magnitude)),
        "nonzero_fraction": float(
            np.mean(magnitude > EPS)
        ),
    }


# ============================================================
# BETA CONSTRUCTION
# ============================================================

def compute_beta(
    D,
    S,
    M,
    O,
    *,
    contrast_robust=False,
    contrast_scale=10.0,
    beta_clip=None,
    null_threshold=None,
    return_components=False,
    verbose=False,
):
    """
    Compute structured agencity field.

    Canonical definition
    --------------------
        beta = J * U

    where:
        J = contrast(D, S)
        U = orientation(M, O)

    Parameters
    ----------
    D :
        Dynamic intensity.

    S :
        Structural intensity.

    M :
        Memory field.

    O :
        Organization field.

    contrast_robust :
        Robust contrast saturation.

    contrast_scale :
        Saturation scale.

    beta_clip :
        Optional clipping magnitude.

    null_threshold :
        Structural null threshold.

    return_components :
        Return additional diagnostics.

    Returns
    -------
    J :
        Structural contrast.

    U :
        Organizational orientation.

    beta :
        Structured agencity field.
    """

    # ========================================================
    # CONTRAST
    # ========================================================

    if verbose:

        print(
            "[beta] "
            "Computing contrast J"
        )

    J = compute_contrast(
        D,
        S,
        robust=contrast_robust,
        robust_scale=contrast_scale,
        verbose=verbose,
    )

    # ========================================================
    # ORIENTATION
    # ========================================================

    if verbose:

        print(
            "[beta] "
            "Computing orientation U"
        )

    U, S_internal = compute_orientation(
        M,
        O,
        null_threshold=null_threshold,
        return_intensity=True,
        verbose=verbose,
    )

    # ========================================================
    # COMBINATION
    # ========================================================

    if verbose:

        print(
            "[beta] "
            "Combining J and U"
        )

    beta_signal = J * U

    # ========================================================
    # OPTIONAL CLIPPING
    # ========================================================

    if beta_clip is not None:

        beta_clip = abs(
            float(beta_clip)
        )

        magnitude = np.abs(
            beta_signal
        )

        mask = magnitude > beta_clip

        if np.any(mask):

            beta_signal = beta_signal.copy()

            beta_signal[mask] = (
                beta_signal[mask]
                * (
                    beta_clip
                    / (
                        magnitude[mask]
                        + EPS
                    )
                )
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    beta_signal = replace_non_finite(
        beta_signal,
        0.0,
    )

    beta_signal = np.asarray(
        beta_signal,
        dtype=complex,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    stats = _safe_beta_stats(
        beta_signal
    )

    if verbose:

        print(
            "[beta] "
            f"|beta| mean="
            f"{stats['mean']:.6f}"
        )

        print(
            "[beta] "
            f"|beta| std="
            f"{stats['std']:.6f}"
        )

        print(
            "[beta] "
            f"|beta| min="
            f"{stats['min']:.6f}"
        )

        print(
            "[beta] "
            f"|beta| max="
            f"{stats['max']:.6f}"
        )

        print(
            "[beta] "
            f"nonzero_fraction="
            f"{stats['nonzero_fraction']:.6f}"
        )

    # ========================================================
    # RETURN
    # ========================================================

    if return_components:

        return {
            "J": J,
            "U": U,
            "S": S_internal,
            "beta": beta_signal,
            "stats": stats,
        }

    return J, U, beta_signal


# ============================================================
# PUBLIC API
# ============================================================

def beta(
    D,
    S,
    M,
    O,
    *,
    contrast_robust=False,
    contrast_scale=10.0,
    beta_clip=None,
    null_threshold=None,
    verbose=False,
):
    """
    Public API returning only beta.
    """

    _, _, out = compute_beta(
        D,
        S,
        M,
        O,
        contrast_robust=contrast_robust,
        contrast_scale=contrast_scale,
        beta_clip=beta_clip,
        null_threshold=null_threshold,
        verbose=verbose,
    )

    return out


def structured_agencity(
    *args,
    **kwargs,
):
    """
    Alias for theoretical compatibility.
    """

    return beta(
        *args,
        **kwargs,
    )