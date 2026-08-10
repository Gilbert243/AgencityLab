"""
multiscale.py

Multi-scale agencity analysis.

Canonical theory
----------------
tau is a structural characteristic time
of the containing system.

This module explores:

    tau_k = k * tau0

without estimating tau from the signal.

The purpose is NOT to redefine the system,
but to study the response of agencity metrics
across structural observation scales.

CRM rule
--------
Canonical regime:
    w = tau

Compressed regime:
    w = tau / A_fact

Compression is activated only when:

    T_obs < 2*tau
"""

from __future__ import annotations

import numpy as np

from .validation import (
    validate_axis,
    validate_signal,
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

from .tau import (
    characteristic_time,
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

from .power import (
    characteristic_power,
)

from .agencity import (
    agencity,
)

from agencitylab.constants.activity_factors import (
    resolve_activity_factor,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _default_scales(
    n=12,
    low=0.5,
    high=2.5,
):
    """
    Default logarithmic scale grid.
    """

    return np.exp(
        np.linspace(
            np.log(low),
            np.log(high),
            n,
        )
    )


def _infer_step(axis):

    diffs = np.diff(axis)

    diffs = diffs[
        np.isfinite(diffs)
        & (np.abs(diffs) > 1e-12)
    ]

    if diffs.size == 0:
        return 1.0

    return float(
        np.median(
            np.abs(diffs)
        )
    )


def _crm_window(
    tau,
    T_obs,
    A_fact,
):
    """
    Determine CRM window mode.
    """

    if T_obs >= 2.0 * tau:

        return {
            "mode": "canonical",
            "window": tau,
        }

    return {
        "mode": "compressed",
        "window": tau / max(A_fact, 1e-12),
    }


# ============================================================
# MULTISCALE ANALYSIS
# ============================================================

def multiscale_agencity(
    t,
    u,
    *,
    scales=None,
    tau="auto",
    P_c="auto",
    A_ref="auto",
    activity_factor="auto",
    domain=None,
    mechanism=None,
    system="generic",
    smooth=False,
    resolution_scale=None,
    return_full=False,
    verbose=False,
):
    """
    Compute agencity metrics
    across multiple structural scales.

    Parameters
    ----------
    t :
        External coordinate axis.

    u :
        Observable signal.

    scales :
        Multiplicative factors applied to tau0.

    tau :
        Structural characteristic time.

    P_c :
        Characteristic power.

    A_ref :
        Canonical normalization scale.

    activity_factor :
        Canonical activity factor.

    domain :
        Physical domain.

    mechanism :
        Physical mechanism.

    system :
        System identifier.

    smooth :
        Apply physical smoothing.

    resolution_scale :
        Observation resolution scale.

    return_full :
        Return detailed outputs.

    Returns
    -------
    dict
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    t = validate_axis(t)

    u = validate_signal(
        u
    ).ravel()

    if scales is None:
        scales = _default_scales()

    scales = np.asarray(
        scales,
        dtype=float,
    )

    if np.any(scales <= 0):
        raise ValueError(
            "all scales must be positive"
        )

    # ========================================================
    # OBSERVATION STRUCTURE
    # ========================================================

    T_obs = float(
        t[-1] - t[0]
    )

    dt = _infer_step(t)

    # ========================================================
    # NORMALIZATION
    # ========================================================

    u_star, A_ref_resolved = normalize_signal(
        u,
        A_ref=A_ref,
        domain=domain,
        method="canonical",
        verbose=False,
    )

    # ========================================================
    # STRUCTURAL PARAMETERS
    # ========================================================

    tau0 = characteristic_time(
        tau=tau,
        system=system,
        domain=domain,
        verbose=False,
    )

    P_c0 = characteristic_power(
        value=None if P_c == "auto" else P_c,
        tau=tau0,
        system=system,
        domain=domain,
        verbose=False,
    )

    A_fact = resolve_activity_factor(
        mechanism=mechanism,
        domain=domain,
        A_fact=activity_factor,
    )

    # ========================================================
    # BASE DYNAMICS
    # ========================================================

    X = activation(
        u_star,
        axis=t,
        verbose=False,
    )

    A = activity(
        X,
        axis=t,
        verbose=False,
    )

    # ========================================================
    # GLOBAL DIAGNOSTICS
    # ========================================================

    if verbose:

        print(
            "[multiscale] "
            f"T_obs={T_obs:.6f}"
        )

        print(
            "[multiscale] "
            f"dt={dt:.6f}"
        )

        print(
            "[multiscale] "
            f"tau0={tau0:.6f}"
        )

        print(
            "[multiscale] "
            f"Pc0={P_c0:.6f}"
        )

        print(
            "[multiscale] "
            f"A_fact={A_fact:.6f}"
        )

        print(
            "[multiscale] "
            f"A_ref={A_ref_resolved:.6f}"
        )

        print(
            "[multiscale] "
            f"scales={scales}"
        )

    # ========================================================
    # STORAGE
    # ========================================================

    taus = []

    windows = []

    modes = []

    resolution_ratios = []

    effective_samples = []

    collapsed_scales = []

    beta_means = []

    beta_stds = []

    J_means = []

    D_means = []

    S_means = []

    M_means = []

    O_means = []

    b_means = []

    theta_stds = []

    crm_coverages = []

    raws = []

    # ========================================================
    # MULTISCALE LOOP
    # ========================================================

    for k in scales:

        tau_k = float(k) * float(tau0)

        crm_cfg = _crm_window(
            tau=tau_k,
            T_obs=T_obs,
            A_fact=A_fact,
        )

        mode = crm_cfg["mode"]

        w_k = crm_cfg["window"]

        resolution_ratio = (
            w_k / max(dt, 1e-12)
        )

        collapsed = (
            resolution_ratio < 2.0
        )

        n_eff = max(
            1,
            int(round(w_k / dt))
        )

        if verbose:

            print(
                "\n[multiscale] "
                f"tau={tau_k:.6f} "
                f"(k={k:.3f})"
            )

            print(
                "[multiscale] "
                f"mode={mode}"
            )

            print(
                "[multiscale] "
                f"window={w_k:.6f}"
            )

            print(
                "[multiscale] "
                f"resolution_ratio="
                f"{resolution_ratio:.6f}"
            )

            print(
                "[multiscale] "
                f"effective_samples="
                f"{n_eff}"
            )

            if collapsed:

                print(
                    "[multiscale warning] "
                    "CRM scale below reliable "
                    "measurement resolution"
                )

        # ====================================================
        # MEMORY
        # ====================================================

        M = memory(
            A,
            tau=tau_k,
            axis=t,
            activity_factor=A_fact,
            mechanism=mechanism,
            domain=domain,
            verbose=False,
        )

        # ====================================================
        # ORGANIZATION
        # ====================================================

        O = organization(
            X,
            tau=tau_k,
            axis=t,
            activity_factor=A_fact,
            mechanism=mechanism,
            domain=domain,
            verbose=False,
        )

        # ====================================================
        # INTENSITIES
        # ====================================================

        D, S = compute_intensities(
            X,
            A,
            M,
            O,
            verbose=False,
        )

        # ====================================================
        # STRUCTURED AGENCITY
        # ====================================================

        J, U, B = compute_beta(
            D,
            S,
            M,
            O,
            verbose=False,
        )

        # ====================================================
        # CHARACTERISTIC POWER
        # ====================================================

        P_ck = characteristic_power(
            value=P_c0,
            tau=tau_k,
            system=system,
            domain=domain,
            verbose=False,
        )

        # ====================================================
        # OBSERVABLE AGENCITY
        # ====================================================

        b = agencity(
            B,
            P_c=P_ck,
            smooth=smooth,
            resolution_scale=resolution_scale,
            verbose=False,
        )

        # ====================================================
        # METRICS
        # ====================================================

        beta_abs = np.abs(B)

        b_abs = np.abs(b)

        theta = np.angle(U)

        crm_valid = np.isfinite(M)

        crm_coverage = float(
            np.mean(crm_valid)
        )

        taus.append(tau_k)

        windows.append(w_k)

        modes.append(mode)

        resolution_ratios.append(
            resolution_ratio
        )

        effective_samples.append(
            n_eff
        )

        collapsed_scales.append(
            collapsed
        )

        beta_means.append(
            float(
                np.nanmean(beta_abs)
            )
        )

        beta_stds.append(
            float(
                np.nanstd(beta_abs)
            )
        )

        J_means.append(
            float(
                np.nanmean(J)
            )
        )

        D_means.append(
            float(
                np.nanmean(D)
            )
        )

        S_means.append(
            float(
                np.nanmean(S)
            )
        )

        M_means.append(
            float(
                np.nanmean(M)
            )
        )

        O_means.append(
            float(
                np.nanmean(O)
            )
        )

        b_means.append(
            float(
                np.nanmean(b_abs)
            )
        )

        theta_stds.append(
            float(
                np.nanstd(theta)
            )
        )

        crm_coverages.append(
            crm_coverage
        )

        # ====================================================
        # RAW STORAGE
        # ====================================================

        if return_full:

            raws.append({

                "scale":
                    float(k),

                "tau":
                    tau_k,

                "window":
                    w_k,

                "mode":
                    mode,

                "resolution_ratio":
                    resolution_ratio,

                "effective_samples":
                    n_eff,

                "collapsed":
                    collapsed,

                "P_c":
                    P_ck,

                "A_fact":
                    A_fact,

                "X":
                    X,

                "A":
                    A,

                "M":
                    M,

                "O":
                    O,

                "D":
                    D,

                "S":
                    S,

                "J":
                    J,

                "U":
                    U,

                "beta":
                    B,

                "b":
                    b,

                "theta":
                    theta,
            })

    # ========================================================
    # GLOBAL OUTPUT
    # ========================================================

    out = {

        # structural
        "tau":
            np.asarray(taus),

        "window":
            np.asarray(windows),

        "mode":
            np.asarray(modes),

        "A_fact":
            float(A_fact),

        "A_ref":
            float(A_ref_resolved),

        "P_c":
            float(P_c0),

        # resolution
        "resolution_ratio":
            np.asarray(
                resolution_ratios
            ),

        "effective_samples":
            np.asarray(
                effective_samples
            ),

        "collapsed_scale":
            np.asarray(
                collapsed_scales
            ),

        # beta
        "beta_mean":
            np.asarray(beta_means),

        "beta_std":
            np.asarray(beta_stds),

        # agencity
        "b_mean":
            np.asarray(b_means),

        # structure
        "J_mean":
            np.asarray(J_means),

        "D_mean":
            np.asarray(D_means),

        "S_mean":
            np.asarray(S_means),

        # memory / organization
        "M_mean":
            np.asarray(M_means),

        "O_mean":
            np.asarray(O_means),

        # angular coherence
        "theta_std":
            np.asarray(theta_stds),

        # CRM validity
        "crm_coverage":
            np.asarray(crm_coverages),

        # metadata
        "domain":
            domain,

        "mechanism":
            mechanism,

        "system":
            system,

        "T_obs":
            T_obs,

        "dt":
            dt,
    }

    # ========================================================
    # OPTIONAL RAWS
    # ========================================================

    if return_full:

        out["raw"] = raws

    # ========================================================
    # FINAL DIAGNOSTICS
    # ========================================================

    if verbose:

        n_collapsed = int(
            np.sum(collapsed_scales)
        )

        print(
            "\n[multiscale] "
            "completed"
        )

        print(
            "[multiscale] "
            f"collapsed_scales="
            f"{n_collapsed}/"
            f"{len(scales)}"
        )

    return out