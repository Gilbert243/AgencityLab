"""
activity_factors.py

Canonical activity factors for AgencityLab.

A_fact represents the efficiency of the active
organizational mechanism.

Canonical CRM memory window:
    w = tau / A_fact

Constraints
-----------
A_fact >= 1

Important
---------
A_fact is NOT estimated from the observable signal u(t).

It is resolved from:
    - physical mechanism,
    - domain,
    - system structure,
    - canonical registries.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    from agencitylab.core.safeguards import ensure_positive
except Exception:
    def ensure_positive(x, minimum=1e-12):
        return max(float(x), minimum)


# ============================================================
# CANONICAL REGISTRIES
# ============================================================

CANONICAL_ACTIVITY_BY_MECHANISM: dict[str, float] = {
    "passive": 1.0,
    "diffusion": 1.0,
    "thermal_conduction": 1.0,

    "natural_convection": 10.0,
    "forced_convection": 100.0,

    "laminar_flow": 5.0,
    "turbulence": 1000.0,

    "oscillator": 10.0,
    "resonant_oscillator": 100.0,

    "biological_cycle": 20.0,
}

CANONICAL_ACTIVITY_BY_DOMAIN: dict[str, float] = {
    "thermal": 10.0,
    "mechanics": 5.0,
    "electronics": 20.0,
    "fluidics": 50.0,
    "biology": 20.0,
}

CANONICAL_ACTIVITY_BY_SYSTEM: dict[str, float] = {
    "generic": 1.0,

    "thermal_water_1l": 10.0,

    "pendulum_small": 10.0,

    "resonant_cavity": 100.0,

    "turbulent_pipe": 500.0,
}

CANONICAL_ACTIVITY_BY_DOMAIN_MECHANISM: dict[tuple[str, str], float] = {
    ("thermal", "natural_convection"): 10.0,
    ("thermal", "forced_convection"): 100.0,

    ("mechanics", "oscillator"): 10.0,
    ("mechanics", "resonant_oscillator"): 100.0,

    ("fluidics", "turbulence"): 1000.0,
}


# ============================================================
# INTERNAL UTILS
# ============================================================

def _norm(x: Any) -> str:

    if x is None:
        return ""

    return str(x).strip().lower()


# ============================================================
# REGISTRATION
# ============================================================

def register_activity_factor(
    key,
    value,
    *,
    scope="mechanism",
):
    """
    Register or override canonical activity factors.

    Parameters
    ----------
    key :
        Registry key.

    value :
        Positive activity factor.

    scope :
        - "mechanism"
        - "domain"
        - "system"
        - "domain_mechanism"
    """

    value = ensure_positive(value)

    scope = _norm(scope)

    # ========================================================
    # MECHANISM
    # ========================================================

    if scope == "mechanism":

        CANONICAL_ACTIVITY_BY_MECHANISM[
            _norm(key)
        ] = value

        return

    # ========================================================
    # DOMAIN
    # ========================================================

    if scope == "domain":

        CANONICAL_ACTIVITY_BY_DOMAIN[
            _norm(key)
        ] = value

        return

    # ========================================================
    # SYSTEM
    # ========================================================

    if scope == "system":

        CANONICAL_ACTIVITY_BY_SYSTEM[
            _norm(key)
        ] = value

        return

    # ========================================================
    # DOMAIN + MECHANISM
    # ========================================================

    if scope == "domain_mechanism":

        if (
            not isinstance(key, (tuple, list))
            or len(key) != 2
        ):
            raise ValueError(
                "Expected (domain, mechanism)"
            )

        d, m = key

        CANONICAL_ACTIVITY_BY_DOMAIN_MECHANISM[
            (_norm(d), _norm(m))
        ] = value

        return

    raise ValueError(
        "scope must be: "
        "mechanism | domain | system | domain_mechanism"
    )


# ============================================================
# RESOLUTION
# ============================================================

def resolve_activity_factor(
    *,
    mechanism=None,
    domain=None,
    system_type=None,
    A_fact=None,
    default=1.0,
):
    """
    Resolve canonical activity factor.

    Resolution priority
    -------------------
    1) explicit override
    2) (domain, mechanism)
    3) system_type
    4) mechanism
    5) domain
    6) default
    """

    # ========================================================
    # EXPLICIT OVERRIDE
    # ========================================================

    if A_fact is not None:

        token = _norm(A_fact)

        if token not in {
            "auto",
            "canonical",
            "default",
        }:
            return ensure_positive(A_fact)

    mechanism_key = _norm(mechanism)

    domain_key = _norm(domain)

    system_key = _norm(system_type)

    # ========================================================
    # DOMAIN + MECHANISM
    # ========================================================

    if domain_key and mechanism_key:

        val = (
            CANONICAL_ACTIVITY_BY_DOMAIN_MECHANISM
            .get(
                (domain_key, mechanism_key)
            )
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # SYSTEM
    # ========================================================

    if system_key:

        val = (
            CANONICAL_ACTIVITY_BY_SYSTEM
            .get(system_key)
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # MECHANISM
    # ========================================================

    if mechanism_key:

        val = (
            CANONICAL_ACTIVITY_BY_MECHANISM
            .get(mechanism_key)
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # DOMAIN
    # ========================================================

    if domain_key:

        val = (
            CANONICAL_ACTIVITY_BY_DOMAIN
            .get(domain_key)
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # DEFAULT
    # ========================================================

    return ensure_positive(default)


# ============================================================
# METADATA HELPERS
# ============================================================

def activity_context_from_metadata(
    metadata: Mapping[str, Any] | None,
):
    """
    Extract activity-factor context from metadata.
    """

    if metadata is None:
        return {}

    return {
        "mechanism": metadata.get("mechanism"),
        "domain": metadata.get("domain"),
        "system_type": metadata.get("system_type"),
        "A_fact": metadata.get("activity_factor"),
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "CANONICAL_ACTIVITY_BY_MECHANISM",
    "CANONICAL_ACTIVITY_BY_DOMAIN",
    "CANONICAL_ACTIVITY_BY_SYSTEM",
    "CANONICAL_ACTIVITY_BY_DOMAIN_MECHANISM",

    "register_activity_factor",

    "resolve_activity_factor",

    "activity_context_from_metadata",
]