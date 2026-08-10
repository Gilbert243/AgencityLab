"""
fundamental_noise_experiment.py

Goal
----
Verify whether Agencity structural quantities naturally
tend toward zero for pure white noise when the number
of samples increases.

Theory
------
If the theory is structurally correct:

    N -> infinity

should imply:

    CRM -> 0
    M   -> 0
    O   -> 0
    S   -> 0
    beta -> 0

for pure white noise.
"""

from __future__ import annotations

import numpy as np

from agencitylab.core.normalization import normalize_signal

from agencitylab.core.activation import (
    activation,
)

from agencitylab.core.activity import (
    activity,
)

from agencitylab.core.crm import (
    causal_moving_correlation,
)

from agencitylab.core.memory import (
    memory,
)

from agencitylab.core.organization import (
    organization,
)

from agencitylab.core.intensity import (
    compute_intensities,
)

from agencitylab.core.beta import (
    compute_beta,
)

from agencitylab.core.orientation import (
    compute_orientation,
)


# ============================================================
# CONFIG
# ============================================================

SIZES = [
    200,
    2_000,
    20_000,
    200_000,
]

TAU = 1.0

DOMAIN = "generic"

MECHANISM = "passive"

RNG = np.random.default_rng(42)


# ============================================================
# HELPERS
# ============================================================

def summarize(name, x):

    x = np.asarray(x)

    return {
        "name": name,
        "mean_abs": float(
            np.nanmean(np.abs(x))
        ),
        "std": float(
            np.nanstd(x)
        ),
        "min": float(
            np.nanmin(x)
        ),
        "max": float(
            np.nanmax(x)
        ),
    }


def print_summary(stats):

    print(
        f"{stats['name']:12s} | "
        f"mean(|.|)={stats['mean_abs']:.6e} | "
        f"std={stats['std']:.6e} | "
        f"min={stats['min']:.6e} | "
        f"max={stats['max']:.6e}"
    )


# ============================================================
# EXPERIMENT
# ============================================================

print("\n")
print("=" * 80)
print("PURE WHITE NOISE FUNDAMENTAL EXPERIMENT")
print("=" * 80)

for N in SIZES:

    print("\n")
    print("-" * 80)
    print(f"N = {N}")
    print("-" * 80)

    # ========================================================
    # AXIS
    # ========================================================

    t = np.linspace(
        0.0,
        10.0,
        N,
    )

    # ========================================================
    # WHITE NOISE
    # ========================================================

    u = RNG.normal(
        0.0,
        1.0,
        N,
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    u_star, A_ref = normalize_signal(
        u,
        A_ref=1.0,
    )

    # ========================================================
    # ACTIVATION
    # ========================================================

    X = activation(
        u_star,
        t,
    )

    # ========================================================
    # ACTIVITY
    # ========================================================

    A = activity(
        X,
        t,
    )

    # ========================================================
    # CRM
    # ========================================================

    crm_X = causal_moving_correlation(
        X,
        tau=TAU,
        axis=t,
        mechanism=MECHANISM,
        domain=DOMAIN,
    )

    crm_A = causal_moving_correlation(
        A,
        tau=TAU,
        axis=t,
        mechanism=MECHANISM,
        domain=DOMAIN,
    )

    # ========================================================
    # MEMORY / ORGANIZATION
    # ========================================================

    M = memory(
        A,
        tau=TAU,
        axis=t,
        mechanism=MECHANISM,
        domain=DOMAIN,
    )

    O = organization(
        X,
        tau=TAU,
        axis=t,
        mechanism=MECHANISM,
        domain=DOMAIN,
    )

    # ========================================================
    # INTENSITIES
    # ========================================================

    D, S = compute_intensities(
        X,
        A,
        M,
        O,
    )

    # ========================================================
    # BETA
    # ========================================================

    J, U, B = compute_beta(
        D,
        S,
        M,
        O,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_summary(
        summarize("CRM(X)", crm_X)
    )

    print_summary(
        summarize("CRM(A)", crm_A)
    )

    print_summary(
        summarize("M", M)
    )

    print_summary(
        summarize("O", O)
    )

    print_summary(
        summarize("S", S)
    )

    print_summary(
        summarize("|beta|", np.abs(B))
    )
    
    # ========================================================
    # GLOBAL COMPLEX OBSERVABLES
    # ========================================================

    beta_mean_complex = np.mean(B)

    beta_abs_mean = np.abs(
        beta_mean_complex
    )

    beta_mean_abs = np.mean(
        np.abs(B)
    )

    phase_coherence = (
        np.abs(np.mean(U))
    )

    print("\n")
    print("[GLOBAL OBSERVABLES]")

    print(
        f"|mean(beta)|      = "
        f"{beta_abs_mean:.6e}"
    )

    print(
        f"mean(|beta|)      = "
        f"{beta_mean_abs:.6e}"
    )

    print(
        f"phase coherence   = "
        f"{phase_coherence:.6e}"
    )

print("\n")
print("=" * 80)
print("END OF EXPERIMENT")
print("=" * 80)