"""
normalization.py

Canonical normalization utilities for AgencityLab.

Canonical theory
----------------
Observable normalization:

    u* = u / A_ref

where:

    A_ref

is a structural reference amplitude determined from:
    - observable type,
    - physical unit,
    - physical domain,
    - metadata,
    - canonical registries.

IMPORTANT
---------
A_ref is NOT estimated from the signal.

Structural separation
---------------------
Modern Agencity theory separates:

    - observable amplitude,
    - local dynamics,
    - structural memory.

Therefore:

    tau DOES NOT belong to normalization.

tau only intervenes in:
    - CRM structural memory,
    - reduced structural times,
    - multiscale organization analysis.

This prevents artificial amplification of activation
in slow systems.

Experimental normalization
--------------------------
Statistical normalization methods remain available
for exploratory workflows only and are explicitly
marked as non-canonical.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agencitylab.constants.reference_amplitudes import (
    reference_context_from_metadata,
    resolve_reference_amplitude,
)

from .safeguards import (
    EPS,
    ensure_positive,
    replace_non_finite,
)

from .validation import (
    as_float_array,
    validate_signal,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def center_signal(
    signal,
    *,
    axis: int = 0,
):
    """
    Remove mean value along the specified axis.

    WARNING
    -------
    Mean removal is NOT part of the canonical theory.
    """

    arr = as_float_array(signal)

    mean = np.nanmean(
        arr,
        axis=axis,
        keepdims=True,
    )

    return arr - mean


# ============================================================
# EXPERIMENTAL SCALE ESTIMATION
# ============================================================

def compute_reference_scale(
    signal,
    *,
    method: str = "std",
    axis: int = 0,
    eps: float = EPS,
):
    """
    Experimental statistical normalization.

    WARNING
    -------
    This function is NOT canonical.

    Available methods
    -----------------
    std:
        Standard deviation.

    mad:
        Median absolute deviation.

    range:
        Dynamic range.
    """

    arr = as_float_array(signal)

    centered = center_signal(
        arr,
        axis=axis,
    )

    method = (
        str(method)
        .strip()
        .lower()
    )

    # ========================================================
    # RANGE
    # ========================================================

    if method == "range":

        scale = (

            np.nanmax(
                arr,
                axis=axis,
                keepdims=True,
            )

            -

            np.nanmin(
                arr,
                axis=axis,
                keepdims=True,
            )
        )

    # ========================================================
    # STANDARD DEVIATION
    # ========================================================

    elif method == "std":

        scale = np.nanstd(
            centered,
            axis=axis,
            keepdims=True,
        )

    # ========================================================
    # MAD
    # ========================================================

    elif method == "mad":

        med = np.nanmedian(
            arr,
            axis=axis,
            keepdims=True,
        )

        scale = np.nanmedian(
            np.abs(arr - med),
            axis=axis,
            keepdims=True,
        )

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:

        raise ValueError(
            f"Unknown reference scale method "
            f"'{method}'"
        )

    scale = ensure_positive(
        scale,
        minimum=eps,
    )

    return scale


# ============================================================
# CANONICAL NORMALIZATION
# ============================================================

def normalize_signal(
    signal,
    *,
    A_ref=None,
    unit=None,
    observable_kind=None,
    domain=None,
    metadata=None,
    method: str = "canonical",
    center: bool = False,
    axis: int = 0,
    clip=None,
    replace_nan=True,
    verbose: bool = False,
):
    """
    Normalize observable signal.

    Canonical definition
    --------------------
        u* = u / A_ref

    Parameters
    ----------
    signal : ndarray

    A_ref : float or "auto"
        Structural reference amplitude.

    unit : str
        Physical unit.

    observable_kind : str
        Observable category.

    domain : str
        Physical domain.

    metadata : dict or metadata object
        Optional metadata container.

    method : str
        Normalization strategy.

    center : bool
        Remove mean before normalization
        (non-canonical).

    clip : float or None
        Optional clipping magnitude.

    replace_nan : bool
        Replace NaN and infinities.

    Returns
    -------
    u_star : ndarray
        Normalized observable.

    ref : float or ndarray
        Reference amplitude used.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    arr = validate_signal(
        signal,
        name="signal",
    )

    arr = np.asarray(
        arr,
        dtype=float,
    )

    # ========================================================
    # OPTIONAL NAN CLEANUP
    # ========================================================

    if replace_nan:

        arr = replace_non_finite(
            arr,
            default=0.0,
        )

    # ========================================================
    # OPTIONAL CENTERING
    # ========================================================

    if center:

        if verbose:

            print(
                "[normalization] "
                "Centering signal "
                "(non-canonical)"
            )

        arr = center_signal(
            arr,
            axis=axis,
        )

    # ========================================================
    # METADATA RESOLUTION
    # ========================================================

    if metadata is not None:

        ctx = reference_context_from_metadata(
            metadata
        )

        unit = (
            unit
            or ctx.get("unit")
        )

        observable_kind = (
            observable_kind
            or ctx.get("observable_kind")
        )

        domain = (
            domain
            or ctx.get("domain")
        )

        if A_ref is None:

            A_ref = (
                ctx.get("A_ref")
                or ctx.get(
                    "reference_amplitude"
                )
            )

    # ========================================================
    # METHOD
    # ========================================================

    method_key = (
        str(method)
        .strip()
        .lower()
    )

    # ========================================================
    # CANONICAL NORMALIZATION
    # ========================================================

    if method_key in {
        "canonical",
        "a_ref",
        "auto",
        "default",
    }:

        if verbose:

            print(
                "[normalization] "
                "Canonical normalization"
            )

        ref = resolve_reference_amplitude(
            unit=unit,
            observable_kind=observable_kind,
            domain=domain,
            A_ref=A_ref,
            default=1.0,
        )

        ref = ensure_positive(ref)

        u_star = arr / ref

    # ========================================================
    # Z-SCORE
    # ========================================================

    elif method_key == "zscore":

        if verbose:

            print(
                "[normalization] "
                "Z-score normalization "
                "(non-canonical)"
            )

        mean = np.mean(
            arr,
            axis=axis,
            keepdims=True,
        )

        std = np.std(
            arr,
            axis=axis,
            keepdims=True,
        )

        ref = ensure_positive(std)

        u_star = (
            arr - mean
        ) / ref

    # ========================================================
    # MIN-MAX
    # ========================================================

    elif method_key == "minmax":

        if verbose:

            print(
                "[normalization] "
                "Min-max normalization "
                "(non-canonical)"
            )

        min_ = np.min(
            arr,
            axis=axis,
            keepdims=True,
        )

        max_ = np.max(
            arr,
            axis=axis,
            keepdims=True,
        )

        ref = ensure_positive(
            max_ - min_
        )

        u_star = (
            arr - min_
        ) / ref

    # ========================================================
    # CENTERED
    # ========================================================

    elif method_key == "centered":

        if verbose:

            print(
                "[normalization] "
                "Centered normalization "
                "(non-canonical)"
            )

        mean = np.mean(
            arr,
            axis=axis,
            keepdims=True,
        )

        ref = 1.0

        u_star = arr - mean

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:

        raise ValueError(
            f"Unknown normalization method "
            f"'{method}'"
        )

    # ========================================================
    # OPTIONAL CLIPPING
    # ========================================================

    if clip is not None:

        clip = abs(float(clip))

        u_star = np.clip(
            u_star,
            -clip,
            clip,
        )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    u_star = replace_non_finite(
        u_star,
        default=0.0,
    )

    u_star = np.asarray(
        u_star,
        dtype=float,
    )

    ref = ensure_positive(ref)

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        ref_scalar = float(
            np.asarray(ref)
            .reshape(-1)[0]
        )

        print(
            "[normalization] "
            f"method={method_key}"
        )

        print(
            "[normalization] "
            f"A_ref≈{ref_scalar:.6g}"
        )

        print(
            "[normalization] "
            f"u* mean="
            f"{np.nanmean(u_star):.6f}"
        )

        print(
            "[normalization] "
            f"u* std="
            f"{np.nanstd(u_star):.6f}"
        )

        print(
            "[normalization] "
            f"u* min="
            f"{np.nanmin(u_star):.6f}"
        )

        print(
            "[normalization] "
            f"u* max="
            f"{np.nanmax(u_star):.6f}"
        )

    return u_star, ref


# ============================================================
# ALIAS
# ============================================================

def normalize_state(
    *args,
    **kwargs,
):
    """
    External readability alias.
    """

    return normalize_signal(
        *args,
        **kwargs,
    )