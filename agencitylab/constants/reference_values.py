"""
Canonical reference values for AgencityLab.

These parameters define the default operating regime of the system.
They are dimension-aware and aligned with Agencity theory.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# STRUCTURE
# ============================================================

@dataclass(frozen=True)
class ReferenceParameter:
    name: str
    symbol: str
    value: float
    unit: str
    description: str


# ============================================================
# NUMERICAL
# ============================================================

DEFAULT_EPSILON = ReferenceParameter(
    name="Numerical epsilon",
    symbol="ε",
    value=1e-12,
    unit="dimensionless",
    description="Small value to ensure numerical stability",
)

DEFAULT_TAU_THRESHOLD = ReferenceParameter(
    name="Activation threshold",
    symbol="τ",
    value=0.5,
    unit="dimensionless",
    description="Threshold for agencity activation",
)


# ============================================================
# TEMPORAL
# ============================================================

DEFAULT_ACTIVITY_WINDOW = ReferenceParameter(
    name="Activity window",
    symbol="W_a",
    value=1,
    unit="steps",
    description="Temporal window for activity integration",
)

DEFAULT_CRM_WINDOW = ReferenceParameter(
    name="Causal response memory window",
    symbol="W_c",
    value=1,
    unit="steps",
    description="Window for causal memory accumulation",
)


# ============================================================
# TIME SCALE
# ============================================================

DEFAULT_REDUCED_TIME_STEP = ReferenceParameter(
    name="Reduced time step",
    symbol="Δt*",
    value=1.0,
    unit="dimensionless",
    description="Normalized time step",
)


# ============================================================
# INFORMATION / POWER / AGENCITY
# ============================================================

DEFAULT_INFORMATION_SCALE = ReferenceParameter(
    name="Information scale",
    symbol="I₀",
    value=1.0,
    unit="nat",
    description="Reference information unit",
)

DEFAULT_POWER_SCALE = ReferenceParameter(
    name="Power scale",
    symbol="P₀",
    value=1.0,
    unit="W",
    description="Reference power",
)

DEFAULT_BEMWIZ_SCALE = ReferenceParameter(
    name="Agencity scale",
    symbol="B₀",
    value=1.0,
    unit="W·nat (Bz)",
    description="Reference agencity unit",
)


# ============================================================
# REGISTRY
# ============================================================

REFERENCE_VALUES = {
    "epsilon": DEFAULT_EPSILON,
    "tau": DEFAULT_TAU_THRESHOLD,
    "activity_window": DEFAULT_ACTIVITY_WINDOW,
    "crm_window": DEFAULT_CRM_WINDOW,
    "dt_reduced": DEFAULT_REDUCED_TIME_STEP,
    "I0": DEFAULT_INFORMATION_SCALE,
    "P0": DEFAULT_POWER_SCALE,
    "B0": DEFAULT_BEMWIZ_SCALE,
}


# ============================================================
# UTILS
# ============================================================

def get_reference(name: str) -> ReferenceParameter:
    if name not in REFERENCE_VALUES:
        raise KeyError(f"Unknown reference parameter '{name}'")
    return REFERENCE_VALUES[name]


def list_references():
    return list(REFERENCE_VALUES.keys())