"""
characteristic_powers.py

Canonical characteristic powers for AgencityLab.

Pc represents the characteristic energetic scale
of the containing physical system.

Important
---------
Pc is NOT estimated from the observable signal u(t).

Canonical relation:
    Pc = E_ref / tau
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

CANONICAL_POWER_BY_SYSTEM: dict[str, float] = {
    "generic": 1.0,

    "thermal_water_1l": 100.0,

    "thermal_air_small": 1.0,

    "pendulum_small": 0.1,

    "rc_small": 0.01,
}

CANONICAL_POWER_BY_DOMAIN: dict[str, float] = {
    "thermal": 10.0,
    "mechanics": 1.0,
    "electronics": 0.1,
    "fluidics": 5.0,
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

def register_characteristic_power(
    key,
    value,
    *,
    scope="system",
):
    """
    Register or override canonical Pc values.
    """

    value = ensure_positive(value)

    scope = _norm(scope)

    if scope == "system":

        CANONICAL_POWER_BY_SYSTEM[
            _norm(key)
        ] = value

        return

    if scope == "domain":

        CANONICAL_POWER_BY_DOMAIN[
            _norm(key)
        ] = value

        return

    raise ValueError(
        "scope must be: system | domain"
    )


# ============================================================
# RESOLUTION
# ============================================================

def resolve_characteristic_power(
    *,
    system=None,
    domain=None,
    Pc=None,
    default=1.0,
):
    """
    Resolve canonical characteristic power Pc.
    """

    if Pc is not None:

        token = _norm(Pc)

        if token not in {
            "auto",
            "canonical",
            "default",
        }:
            return ensure_positive(Pc)

    system_key = _norm(system)

    domain_key = _norm(domain)

    # ========================================================
    # SYSTEM
    # ========================================================

    if system_key:

        val = (
            CANONICAL_POWER_BY_SYSTEM
            .get(system_key)
        )

        if val is not None:
            return ensure_positive(val)

    # ========================================================
    # DOMAIN
    # ========================================================

    if domain_key:

        val = (
            CANONICAL_POWER_BY_DOMAIN
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

def power_context_from_metadata(
    metadata: Mapping[str, Any] | None,
):
    """
    Extract characteristic power context from metadata.
    """

    if metadata is None:
        return {}

    return {
        "system": metadata.get("system_type"),
        "domain": metadata.get("domain"),
        "Pc": metadata.get("characteristic_power"),
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "CANONICAL_POWER_BY_SYSTEM",
    "CANONICAL_POWER_BY_DOMAIN",

    "register_characteristic_power",

    "resolve_characteristic_power",

    "power_context_from_metadata",
]