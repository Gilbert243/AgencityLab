"""
agencity.py

Observable agencity b(t).

Canonical theory:
    b(t) = P_c * beta(t)

where:
    P_c  = structural characteristic power
    beta = reduced structured agencity

This module provides:
    - observable agencity
    - reduced agencity rate
    - decomposition helpers
    - agencity criteria
    - coherence diagnostics
    - full canonical pipeline

Canonical principles
--------------------
- tau is structural and independent from u(t)
- P_c is structural and independent from u(t)
- A_fact is structural and independent from u(t)
- CRM uses:
      w = tau / A_fact
- normalization uses canonical A_ref

Real-agencity criterion
-----------------------
A system satisfies real agencity if:

    1) S > 0
    2) angular variance Σ_Θ is low
    3) |b| is significant

This criterion is derived from:
    Volume 2, Chapter 7, §7.8
"""

from __future__ import annotations

import numpy as np

from .safeguards import (
    EPS,
    replace_non_finite,
)

from .validation import (
    validate_axis,
    validate_window_size,
)

from .tau import (
    characteristic_time,
)

from .power import (
    characteristic_power,
)

from .coherence import (
    compute_theta,
    circular_variance,
    phase_coherence,
    directional_stability,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _broadcast_power(
    power,
    target_shape,
):
    """
    Broadcast characteristic power
    to target shape.
    """

    power = np.asarray(
        power,
        dtype=float,
    )

    if power.ndim == 0:

        return np.full(
            target_shape,
            float(power),
            dtype=float,
        )

    return np.broadcast_to(
        power,
        target_shape,
    ).astype(
        float,
        copy=False,
    )


def _effective_delta_star(
    delta_star,
    axis_star=None,
):
    """
    Infer effective reduced resolution.
    """

    if axis_star is None:

        return float(
            validate_window_size(
                delta_star
            )
        )

    axis_star = validate_axis(
        axis_star
    )

    diffs = np.diff(axis_star)

    diffs = diffs[
        np.isfinite(diffs)
        & (np.abs(diffs) > EPS)
    ]

    if diffs.size == 0:
        return 1.0

    step = float(
        np.median(
            np.abs(diffs)
        )
    )

    delta_star = float(
        validate_window_size(
            delta_star
        )
    )

    shift = max(
        1,
        int(round(delta_star / step))
    )

    return shift * step


# ============================================================
# AGENCITY
# ============================================================

def agencity(
    beta_signal,
    P_c=1.0,
    *,
    smooth=False,
    resolution_scale=None,
    verbose=False,
):
    """
    Compute observable agencity:

        b = P_c * beta

    Parameters
    ----------
    beta_signal :
        Reduced structured agencity.

    P_c :
        Structural characteristic power.

    smooth :
        Apply physical resolution filtering.

    resolution_scale :
        Physical resolution scale.
    """

    beta_signal = np.asarray(
        beta_signal,
        dtype=(
            complex
            if np.iscomplexobj(beta_signal)
            else float
        ),
    )

    if beta_signal.size == 0:
        raise ValueError(
            "beta_signal cannot be empty"
        )

    # ========================================================
    # POWER
    # ========================================================

    power = _broadcast_power(
        P_c,
        beta_signal.shape,
    )

    # ========================================================
    # RAW AGENCITY
    # ========================================================

    b = power * beta_signal

    # ========================================================
    # OPTIONAL PHYSICAL SMOOTHING
    # ========================================================

    if smooth and resolution_scale is not None:

        scale = int(
            max(
                1,
                round(
                    float(resolution_scale)
                )
            )
        )

        if scale > 1:

            if verbose:

                print(
                    "[agencity] "
                    f"smoothing scale={scale}"
                )

            kernel = np.ones(scale) / scale

            if np.iscomplexobj(b):

                real = np.convolve(
                    np.real(b),
                    kernel,
                    mode="same",
                )

                imag = np.convolve(
                    np.imag(b),
                    kernel,
                    mode="same",
                )

                b = real + 1j * imag

            else:

                b = np.convolve(
                    b,
                    kernel,
                    mode="same",
                )

    # ========================================================
    # SAFEGUARDS
    # ========================================================

    b = replace_non_finite(
        b,
        0.0,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        mag = (
            np.abs(b)
            if np.iscomplexobj(b)
            else b
        )

        print(
            "[agencity] "
            f"|b| mean={np.nanmean(mag):.6f}"
        )

        print(
            "[agencity] "
            f"|b| std={np.nanstd(mag):.6f}"
        )

        print(
            "[agencity] "
            f"|b| max={np.nanmax(mag):.6f}"
        )

    return b


# ============================================================
# REDUCED AGENCITY RATE
# ============================================================

def agencity_rate(
    beta_signal,
    *,
    delta_star=1.0,
    axis_star=None,
    fill_value=np.nan,
    verbose=False,
):
    """
    Compute reduced agencity derivative:

        dbeta / d(xi*)

    Diagnostic helper only.
    """

    beta_signal = np.asarray(
        beta_signal
    ).ravel()

    if beta_signal.size < 2:

        raise ValueError(
            "beta_signal must contain "
            "at least two samples"
        )

    delta_eff = _effective_delta_star(
        delta_star,
        axis_star=axis_star,
    )

    # ========================================================
    # SHIFT
    # ========================================================

    if axis_star is None:

        shift = max(
            1,
            int(round(float(delta_star)))
        )

    else:

        axis_star = validate_axis(
            axis_star,
            expected_length=beta_signal.size,
        )

        diffs = np.diff(axis_star)

        diffs = diffs[
            np.isfinite(diffs)
            & (np.abs(diffs) > EPS)
        ]

        step = (
            float(
                np.median(
                    np.abs(diffs)
                )
            )
            if diffs.size
            else 1.0
        )

        shift = max(
            1,
            int(round(delta_eff / step))
        )

        delta_eff = shift * step

    # ========================================================
    # DERIVATIVE
    # ========================================================

    out = np.full(
        beta_signal.shape,
        fill_value,
        dtype=beta_signal.dtype,
    )

    out[shift:] = (
        beta_signal[shift:]
        - beta_signal[:-shift]
    ) / max(delta_eff, EPS)

    out = replace_non_finite(
        out,
        0.0,
    )

    if verbose:

        print(
            "[agencity_rate] "
            f"mean={np.nanmean(np.abs(out)):.6f}"
        )

    return out


# ============================================================
# DECOMPOSITION
# ============================================================

def decompose_agencity(
    b,
    *,
    verbose=False,
):
    """
    Decompose complex agencity.

    Returns
    -------
    bx :
        Real component.

    by :
        Imaginary component.

    mag :
        Magnitude.
    """

    b = np.asarray(b)

    bx = np.real(b)

    by = np.imag(b)

    mag = np.abs(b)

    if verbose:

        print(
            "[decomposition] "
            f"|b| mean={np.mean(mag):.6f}"
        )

    return bx, by, mag


# ============================================================
# REAL AGENCITY CRITERION
# ============================================================

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
    """
    Evaluate canonical real-agencity criterion.

    Canonical criterion
    -------------------
        Real agencity iff:

            S > 0
            Σ_Θ small
            |b| significant

    Parameters
    ----------
    M :
        Structural memory.

    O :
        Structural organization.

    S :
        Structural intensity.

    b :
        Observable agencity flow.

    Returns
    -------
    dict
    """

    M = np.asarray(
        M,
        dtype=float,
    )

    O = np.asarray(
        O,
        dtype=float,
    )

    S = np.asarray(
        S,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=complex,
    )

    # ========================================================
    # THETA
    # ========================================================

    theta = compute_theta(
        M,
        O,
    )

    sigma_theta = float(
        circular_variance(theta)
    )

    theta_coherence = float(
        phase_coherence(theta)
    )

    theta_stability = float(
        directional_stability(theta)
    )

    # ========================================================
    # STRUCTURAL INTENSITY
    # ========================================================

    mean_S = float(
        np.mean(S)
    )

    # ========================================================
    # AGENCITY FLOW
    # ========================================================

    mean_b = float(
        np.mean(np.abs(b))
    )

    # ========================================================
    # CONDITIONS
    # ========================================================

    has_structure = (
        mean_S > s_threshold
    )

    stable_orientation = (
        sigma_theta
        < theta_variance_threshold
    )

    significant_flow = (
        mean_b > b_threshold
    )

    # ========================================================
    # GLOBAL DECISION
    # ========================================================

    real_agencity = (
        has_structure
        and stable_orientation
        and significant_flow
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = (
        int(has_structure)
        + int(stable_orientation)
        + int(significant_flow)
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    out = {

        # criterion
        "real_agencity":
            bool(real_agencity),

        # conditions
        "has_structure":
            bool(has_structure),

        "stable_orientation":
            bool(stable_orientation),

        "significant_flow":
            bool(significant_flow),

        # scores
        "score":
            int(score),

        "score_normalized":
            float(score / 3.0),

        # structural quantities
        "mean_S":
            mean_S,

        "sigma_theta":
            sigma_theta,

        "theta_coherence":
            theta_coherence,

        "theta_stability":
            theta_stability,

        "mean_abs_b":
            mean_b,
    }

    if verbose:

        print(
            "[agencity criteria] "
            f"real_agencity="
            f"{real_agencity}"
        )

        print(
            "[agencity criteria] "
            f"score={score}/3"
        )

        print(
            "[agencity criteria] "
            f"mean_S={mean_S:.6f}"
        )

        print(
            "[agencity criteria] "
            f"sigma_theta={sigma_theta:.6f}"
        )

        print(
            "[agencity criteria] "
            f"|b|={mean_b:.6f}"
        )

    return out


# ============================================================
# FULL CANONICAL PIPELINE
# ============================================================

def compute_full_agencity(
    t,
    u,
    *,
    tau="auto",
    P_c="auto",
    A_ref="auto",
    activity_factor="auto",
    domain=None,
    mechanism=None,
    system="generic",
    resolution_scale=None,
    smooth=False,
    verbose=False,
):
    """
    Full canonical pipeline.

    Pipeline
    --------
        u
        -> u*
        -> X*
        -> A*
        -> CRM
        -> M,O
        -> D,S
        -> beta
        -> b
    """

    if verbose:

        print(
            "\n[full] "
            "Starting pipeline\n"
        )

    from .normalization import (
        normalize_signal,
    )

    from .activation import (
        activation,
    )

    from .activity import (
        activity,
    )

    from .memory import (
        memory,
    )

    from .organization import (
        organization,
    )

    from .intensity import (
        compute_intensities,
    )

    from .beta import (
        compute_beta,
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    u_star, A_ref_resolved = normalize_signal(
        u,
        A_ref=A_ref,
        domain=domain,
        method="canonical",
        verbose=verbose,
    )

    # ========================================================
    # STRUCTURAL PARAMETERS
    # ========================================================

    tau = characteristic_time(
        tau=tau,
        system=system,
        domain=domain,
        verbose=verbose,
    )

    P_c = characteristic_power(
        value=None if P_c == "auto" else P_c,
        system=system,
        domain=domain,
        tau=tau,
        verbose=verbose,
    )

    # ========================================================
    # ACTIVATION
    # ========================================================

    X = activation(
        u_star,
        axis=t,
        verbose=verbose,
    )

    # ========================================================
    # ACTIVITY
    # ========================================================

    A = activity(
        X,
        axis=t,
        verbose=verbose,
    )

    # ========================================================
    # MEMORY
    # ========================================================

    M = memory(
        A,
        tau=tau,
        axis=t,
        activity_factor=activity_factor,
        mechanism=mechanism,
        domain=domain,
        verbose=verbose,
    )

    # ========================================================
    # ORGANIZATION
    # ========================================================

    O = organization(
        X,
        tau=tau,
        axis=t,
        activity_factor=activity_factor,
        mechanism=mechanism,
        domain=domain,
        verbose=verbose,
    )

    # ========================================================
    # INTENSITIES
    # ========================================================

    D, S = compute_intensities(
        X,
        A,
        M,
        O,
        verbose=verbose,
    )

    # ========================================================
    # STRUCTURED AGENCITY
    # ========================================================

    J, U, B = compute_beta(
        D,
        S,
        M,
        O,
        verbose=verbose,
    )

    # ========================================================
    # OBSERVABLE AGENCITY
    # ========================================================

    b = agencity(
        B,
        P_c=P_c,
        smooth=smooth,
        resolution_scale=resolution_scale,
        verbose=verbose,
    )

    # ========================================================
    # REAL AGENCITY CRITERION
    # ========================================================

    criteria = agencity_criteria(
        M,
        O,
        S,
        b,
        verbose=verbose,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    out = {

        # normalized
        "u_star":
            u_star,

        # scales
        "A_ref":
            A_ref_resolved,

        "tau":
            tau,

        "P_c":
            P_c,

        "A_fact":
            activity_factor,

        # dynamics
        "X":
            X,

        "A":
            A,

        # organization
        "M":
            M,

        "O":
            O,

        # intensities
        "D":
            D,

        "S":
            S,

        # structure
        "J":
            J,

        "U":
            U,

        # reduced agencity
        "beta":
            B,

        # observable agencity
        "b":
            b,

        # agencity criterion
        "criteria":
            criteria,
    }

    if verbose:

        print(
            "\n[full] "
            "Pipeline completed\n"
        )

    return out