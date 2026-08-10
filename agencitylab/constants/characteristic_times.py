"""
characteristic_times.py

Canonical characteristic times for AgencityLab.

tau represents the structural characteristic time
of the containing physical system.

Important
---------
tau is NOT estimated from the observable signal u(t).

It must come from:
    - physical laws,
    - geometry,
    - inertia,
    - thermal capacity,
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
# REGISTRIES
# ============================================================

CANONICAL_TAU_BY_SYSTEM: dict[str, float] = {
    "generic": 1.0,

    "thermal_water_1l": 5000.0,

    "thermal_air_small": 100.0,

    "pendulum_small": 2.0,

    "rc_small": 0.01,

    "rl_small": 0.05,
}

CANONICAL_TAU_BY_DOMAIN: dict[str, float] = {
    "thermal": 1000.0,
    "mechanics": 1.0,
    "electronics": 0.01,
    "fluidics": 100.0,
}


# ============================================================
# INTERNAL
# ============================================================

def _norm(x: Any) -> str:

    if x is None:
        return ""

    return str(x).strip().lower()


# ============================================================
# REGISTRATION
# ============================================================

def register_characteristic_time(
    key,
    value,
    *,
    scope="system",
):
    """
    Register or override canonical tau values.
    """

    value = ensure_positive(value)

    scope = _norm(scope)

    if scope == "system":

        CANONICAL_TAU_BY_SYSTEM[
            _norm(key)
        ] = value

        return

    if scope == "domain":

        CANONICAL_TAU_BY_DOMAIN[
            _norm(key)
        ] = value

        return

    raise ValueError(
        "scope must be: system | domain"
    )


# ============================================================
# RESOLUTION
# ============================================================

def resolve_characteristic_time(
    *,
    system=None,
    domain=None,
    tau=None,
    default=1.0,
):
    """
    Resolve canonical characteristic time tau.
    """

    if tau is not None:

        token = _norm(tau)

        if token not in {
            "auto",
            "canonical",
            "default",
        }:
            return ensure_positive(tau)

    system_key = _norm(system)

    domain_key = _norm(domain)

    # ========================================================
    # SYSTEM
    # ========================================================

    if system_key:

        val = (
            CANONICAL_TAU_BY_SYSTEM
            .get(system_key)
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # DOMAIN
    # ========================================================

    if domain_key:

        val = (
            CANONICAL_TAU_BY_DOMAIN
            .get(domain_key)
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # DEFAULT
    # ========================================================

    return ensure_positive(default)


# ============================================================
# METADATA
# ============================================================

def tau_context_from_metadata(
    metadata: Mapping[str, Any] | None,
):
    """
    Extract tau context from metadata.
    """

    if metadata is None:
        return {}

    return {
        "system": metadata.get("system_type"),
        "domain": metadata.get("domain"),
        "tau": metadata.get("characteristic_time"),
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "CANONICAL_TAU_BY_SYSTEM",
    "CANONICAL_TAU_BY_DOMAIN",

    "register_characteristic_time",

    "resolve_characteristic_time",

    "tau_context_from_metadata",
]