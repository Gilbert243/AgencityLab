"""Characteristic-power registry for AgencityLab.

``P_c`` is a structural energetic capacity of the containing system. The canonical
resolver never derives it from the observed signal and has no arbitrary generic fallback.
The accepted domain is finite ``P_c >= 0``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CANONICAL_POWER_BY_SYSTEM: dict[str, float] = {}
CANONICAL_POWER_BY_DOMAIN: dict[str, float] = {}


def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _nonnegative_power(value, *, name="P_c") -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return out


def register_characteristic_power(key, value, *, scope="system"):
    """Register an explicit physical characteristic-power convention."""
    value = _nonnegative_power(value)
    scope = _norm(scope)
    if scope == "system":
        CANONICAL_POWER_BY_SYSTEM[_norm(key)] = value
        return
    if scope == "domain":
        CANONICAL_POWER_BY_DOMAIN[_norm(key)] = value
        return
    raise ValueError("scope must be: system | domain")


def resolve_characteristic_power(*, system=None, domain=None, Pc=None, default=None):
    """Resolve finite ``P_c >= 0`` without signal-derived or epsilon fallback."""
    if Pc is not None and _norm(Pc) not in {"auto", "canonical", "default"}:
        return _nonnegative_power(Pc)

    system_key = _norm(system)
    domain_key = _norm(domain)
    if system_key and system_key in CANONICAL_POWER_BY_SYSTEM:
        return CANONICAL_POWER_BY_SYSTEM[system_key]
    if domain_key and domain_key in CANONICAL_POWER_BY_DOMAIN:
        return CANONICAL_POWER_BY_DOMAIN[domain_key]
    if default is not None:
        return _nonnegative_power(default)

    raise ValueError(
        "P_c is a structural physical parameter; provide it explicitly, derive it from "
        "documented container energetics, or register a physical convention"
    )


def power_context_from_metadata(metadata: Mapping[str, Any] | None):
    """Extract characteristic-power context from metadata."""
    if metadata is None:
        return {}
    return {
        "system": metadata.get("system_type"),
        "domain": metadata.get("domain"),
        "Pc": metadata.get("characteristic_power"),
    }


__all__ = [
    "CANONICAL_POWER_BY_SYSTEM",
    "CANONICAL_POWER_BY_DOMAIN",
    "register_characteristic_power",
    "resolve_characteristic_power",
    "power_context_from_metadata",
]
