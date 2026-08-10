"""
test.py

Canonical AgencityLab pipeline test.

This script demonstrates:

    u(t)
    → normalization
    → activation
    → activity
    → CRM
    → memory
    → organization
    → intensities
    → beta
    → agencity

using the stabilized Agencity theory:

    - tau is structural
    - Pc is structural
    - A_fact is canonical
"""

from __future__ import annotations

import numpy as np

from agencitylab.core import (

    # dynamics
    activation,
    activity,

    # temporal
    characteristic_time,
    causal_moving_correlation,

    # structure
    memory,
    organization,

    # intensities
    compute_intensities,
    compute_contrast,

    # orientation
    compute_orientation,
    compute_angle,

    # beta
    compute_beta,

    # power
    characteristic_power,

    # agencity
    agencity,
    agencity_rate,
    decompose_agencity,
    agencity_criteria,

    # high-level
    compute_full_agencity,
    multiscale_agencity,
)

from agencitylab.analysis.signature import (
    agencity_signature,
)


# ============================================================
# SIGNAL
# ============================================================

np.random.seed(42)

t = np.linspace(0.0, 10.0, 200)

u = (
    np.sin(2.0 * np.pi * 0.5 * t)
    + 0.2 * np.random.randn(len(t))
)

print("\n==============================")
print("SIGNAL")
print("==============================")

print("u mean:", np.mean(u))
print("u std :", np.std(u))


# ============================================================
# STRUCTURAL PARAMETERS
# ============================================================

print("\n==============================")
print("STRUCTURAL PARAMETERS")
print("==============================")

tau = characteristic_time(
    system="generic",
    domain="mechanics",
    verbose=True,
)

P_c = characteristic_power(
    system="generic",
    domain="mechanics",
    tau=tau,
    verbose=True,
)

print("tau =", tau)
print("Pc   =", P_c)


# ============================================================
# ACTIVATION
# ============================================================

print("\n==============================")
print("ACTIVATION")
print("==============================")

X = activation(
    u,
    axis=t,
    verbose=True,
)

print("X mean:", np.nanmean(X))


# ============================================================
# ACTIVITY
# ============================================================

print("\n==============================")
print("ACTIVITY")
print("==============================")

A = activity(
    X,
    axis=t,
    verbose=True,
)

print("A mean:", np.nanmean(A))

print("A max:", np.max(np.abs(A)))
print("X max:", np.max(np.abs(X)))

# ============================================================
# CRM
# ============================================================

print("\n==============================")
print("CRM")
print("==============================")

crm = causal_moving_correlation(
    X,
    tau,
    axis=t,
    domain="mechanics",
    mechanism="oscillator",
    verbose=True,
)

print("crm mean:", np.nanmean(crm))


# ============================================================
# MEMORY
# ============================================================

print("\n==============================")
print("MEMORY")
print("==============================")

M = memory(
    A,
    tau,
    axis=t,
    verbose=True,
)

print("M mean:", np.nanmean(M))


# ============================================================
# ORGANIZATION
# ============================================================

print("\n==============================")
print("ORGANIZATION")
print("==============================")

O = organization(
    X,
    tau,
    axis=t,
    verbose=True,
)

print("O mean:", np.nanmean(O))


# ============================================================
# INTENSITIES
# ============================================================

print("\n==============================")
print("INTENSITIES")
print("==============================")

D, S = compute_intensities(
    X,
    A,
    M,
    O,
    verbose=True,
)

print("D mean:", np.nanmean(D))
print("S mean:", np.nanmean(S))


# ============================================================
# CONTRAST
# ============================================================

print("\n==============================")
print("CONTRAST")
print("==============================")

J = compute_contrast(
    D,
    S,
    verbose=True,
)

print("J mean:", np.nanmean(J))


# ============================================================
# ORIENTATION
# ============================================================

print("\n==============================")
print("ORIENTATION")
print("==============================")

U = compute_orientation(
    M,
    O,
    verbose=True,
)

theta = compute_angle(
    U,
    verbose=True,
)

print("|U| mean:", np.nanmean(np.abs(U)))


# ============================================================
# BETA
# ============================================================

print("\n==============================")
print("BETA")
print("==============================")

J2, U2, B = compute_beta(
    D,
    S,
    M,
    O,
    verbose=True,
)

print("|B| mean:", np.nanmean(np.abs(B)))


# ============================================================
# AGENCITY
# ============================================================

print("\n==============================")
print("AGENCITY")
print("==============================")

b = agencity(
    B,
    P_c,
    verbose=True,
)

rate = agencity_rate(
    B,
    axis_star=t,
    verbose=True,
)

print("|b| mean:", np.nanmean(np.abs(b)))
print("rate mean:", np.nanmean(np.abs(rate)))


# ============================================================
# DECOMPOSITION
# ============================================================

print("\n==============================")
print("DECOMPOSITION")
print("==============================")

bx, by, mag = decompose_agencity(
    b,
    verbose=True,
)

print("bx mean:", np.nanmean(bx))
print("by mean:", np.nanmean(by))
print("|b| mean:", np.nanmean(mag))


# ============================================================
# CRITERIA
# ============================================================

print("\n==============================")
print("CRITERIA")
print("==============================")

crit = agencity_criteria(
    B,
    J,
    theta,
    verbose=True,
)

print(crit)


# ============================================================
# FULL PIPELINE
# ============================================================

print("\n==============================")
print("FULL PIPELINE")
print("==============================")

res = compute_full_agencity(
    t,
    u,
    tau=tau,
    P_c=P_c,
    domain="mechanics",
    mechanism="oscillator",
    verbose=True,
)

print("full pipeline keys:")
print(sorted(res.keys()))


# ============================================================
# MULTISCALE
# ============================================================

print("\n==============================")
print("MULTISCALE")
print("==============================")

ms = multiscale_agencity(
    t,
    u,
    tau=tau,
    P_c=P_c,
    domain="mechanics",
    mechanism="oscillator",
    verbose=True,
)

print("taus:", ms["tau"])
print("beta_mean:", ms["beta_mean"])


# ============================================================
# SIGNATURE
# ============================================================

print("\n==============================")
print("SIGNATURE")
print("==============================")

sig = agencity_signature(
    ms["tau"],
    ms["beta_mean"],
    verbose=True,
)

print("signature slope:", sig["slope"])
print("signature regime:", sig["regime"])


print("\n==============================")
print("TEST COMPLETED")
print("==============================")