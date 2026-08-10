"""
crm.py

Causal moving correlation (CRM).

Canonical theory
----------------
Canonical regime:
    w = tau

Compressed regime:
    w = tau / A_fact

CRM definition:
    CRM(t) = Corr([t-w,t], [t-2w,t-w])

where:
    tau   = structural characteristic time
    A_fact >= 1 = organizational activity factor

Interpretation
--------------
tau :
    inertial structural time of the containing system.

A_fact :
    activity acceleration factor associated with
    organized physical mechanisms.

Examples:
    convection  -> Nusselt number
    oscillator  -> quality factor Q
    turbulence  -> Reynolds number

Important
---------
tau and A_fact are structural parameters and must remain
independent from the observable signal u(t).

Canonical rule
--------------
If the observation duration is sufficient:

    T_obs >= 2*tau

then:
    w = tau

Otherwise:
    w = tau / A_fact

This preserves the original CRM definition whenever possible,
and only activates compressed memory when the experiment is
too short compared to the structural timescale.
"""

from __future__ import annotations

import numpy as np

from .validation import (
    validate_signal,
    validate_axis,
    validate_positive_scalar,
)

from .safeguards import (
    EPS,
    ensure_positive,
    replace_non_finite,
)

from agencitylab.constants.activity_factors import (
    resolve_activity_factor,
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_CRM_SAMPLES = 10


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


def _window_to_samples(
    window,
    axis,
):
    """
    Convert physical window to sample count.
    """

    step = _infer_step(axis)

    n = int(
        round(window / step)
    )

    return max(
        MIN_CRM_SAMPLES,
        n,
    )


def _pearson_corr(a, b):
    """
    Stable Pearson correlation.
    """

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    a = replace_non_finite(
        a,
        default=0.0,
    )

    b = replace_non_finite(
        b,
        default=0.0,
    )

    a = a - np.mean(a)

    b = b - np.mean(b)

    var_a = np.mean(a * a)

    var_b = np.mean(b * b)

    if var_a < EPS or var_b < EPS:
        return 0.0

    cov = np.mean(a * b)

    return float(
        cov / (
            np.sqrt(var_a * var_b)
            + EPS
        )
    )


def _observation_duration(axis):
    """
    Observation duration.
    """

    return float(
        axis[-1] - axis[0]
    )


# ============================================================
# CRM
# ============================================================

def causal_moving_correlation(
    signal,
    tau,
    *,
    axis,
    activity_factor="auto",
    mechanism=None,
    domain=None,
    system_type=None,
    force_compressed=False,
    verbose=False,
):
    """
    Compute causal moving correlation (CRM).

    Canonical definition
    --------------------
        CRM(t) =
            Corr(
                [t-w, t],
                [t-2w, t-w]
            )

    Window selection
    ----------------

    Canonical mode:
        w = tau

    Compression mode:
        w = tau / A_fact

    Compression is activated only when:

        T_obs < 2*tau

    or if:
        force_compressed=True

    Parameters
    ----------
    signal :
        Input signal.

    tau :
        Structural characteristic time.

    axis :
        Observation coordinate.

    activity_factor :
        Organizational activity factor.

    mechanism :
        Dominant physical mechanism.

    domain :
        Scientific domain.

    system_type :
        Structural system category.

    force_compressed :
        Force compressed CRM mode.

    verbose :
        Enable diagnostics.

    Returns
    -------
    crm : ndarray
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    x = validate_signal(
        signal,
        name="signal",
    ).ravel()

    axis = validate_axis(
        axis,
        expected_length=len(x),
        name="axis",
    )

    tau = validate_positive_scalar(
        tau,
        name="tau",
    )

    # ========================================================
    # STRUCTURAL PARAMETERS
    # ========================================================

    A_fact = resolve_activity_factor(
        mechanism=mechanism,
        domain=domain,
        A_fact=activity_factor,
    )

    A_fact = ensure_positive(
        A_fact,
    )

    # ========================================================
    # OBSERVATION DURATION
    # ========================================================

    T_obs = _observation_duration(
        axis
    )

    # ========================================================
    # WINDOW MODE
    # ========================================================

    if (
        T_obs >= 2.0 * tau
        and not force_compressed
    ):

        mode = "canonical"

        w = tau

    else:

        mode = "compressed"

        w = tau / A_fact

    w = ensure_positive(w)

    # ========================================================
    # RESOLUTION ANALYSIS
    # ========================================================

    step = _infer_step(axis)

    resolution_ratio = (
        w / max(step, EPS)
    )

    # ========================================================
    # WINDOW SIZE
    # ========================================================

    n = _window_to_samples(
        w,
        axis,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            f"[crm] mode={mode}"
        )

        print(
            f"[crm] tau={tau}"
        )

        print(
            f"[crm] T_obs={T_obs}"
        )

        print(
            f"[crm] A_fact={A_fact}"
        )

        print(
            f"[crm] effective window={w}"
        )

        print(
            f"[crm] resolution step={step}"
        )

        print(
            "[crm] "
            f"resolution ratio={resolution_ratio:.6f}"
        )

        print(
            f"[crm] window samples={n}"
        )

        if resolution_ratio < 2.0:

            print(
                "[crm warning] "
                "effective window below "
                "reliable measurement resolution"
            )

            print(
                "[crm warning] "
                "multiscale collapse may occur"
            )

    # ========================================================
    # MINIMUM LENGTH
    # ========================================================

    if len(x) < 2 * n:

        raise ValueError(
            "signal too short for CRM window"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    out = np.full_like(
        x,
        np.nan,
        dtype=float,
    )

    # ========================================================
    # MOVING CORRELATION
    # ========================================================

    for end in range(
        2 * n - 1,
        len(x),
    ):

        recent = x[
            end - n + 1 :
            end + 1
        ]

        previous = x[
            end - 2 * n + 1 :
            end - n + 1
        ]

        out[end] = _pearson_corr(
            recent,
            previous,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    out = replace_non_finite(
        out,
        default=0.0,
    )

    # ========================================================
    # FINAL DIAGNOSTICS
    # ========================================================

    if verbose:

        valid = np.isfinite(out)

        if np.any(valid):

            print(
                "[crm] "
                f"mean={np.nanmean(out):.6f}"
            )

            print(
                "[crm] "
                f"std={np.nanstd(out):.6f}"
            )

            print(
                "[crm] "
                f"min={np.nanmin(out):.6f}"
            )

            print(
                "[crm] "
                f"max={np.nanmax(out):.6f}"
            )

    return out


# ============================================================
# EXPLICIT ALIAS
# ============================================================

def crm_tau(
    signal,
    tau,
    *,
    axis,
    activity_factor="auto",
    mechanism=None,
    domain=None,
    system_type=None,
    force_compressed=False,
    verbose=False,
):
    """
    Explicit alias for canonical CRM theory.
    """

    return causal_moving_correlation(
        signal,
        tau=tau,
        axis=axis,
        activity_factor=activity_factor,
        mechanism=mechanism,
        domain=domain,
        system_type=system_type,
        force_compressed=force_compressed,
        verbose=verbose,
    )