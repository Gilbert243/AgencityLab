"""Structural characteristic-time registry for AgencityLab.

``tau`` is a physical property of the containing system. The canonical resolver accepts
an explicit value or a deliberately registered physical convention. It never estimates
``tau`` from the observed signal and has no arbitrary generic fallback.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CANONICAL_TAU_BY_SYSTEM: dict[str, float] = {}
CANONICAL_TAU_BY_DOMAIN: dict[str, float] = {}


def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _positive(value, *, name="tau") -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return out


def register_characteristic_time(key, value, *, scope="system"):
    """Register an explicit physical ``tau`` convention."""
    value = _positive(value)
    scope = _norm(scope)
    if scope == "system":
        CANONICAL_TAU_BY_SYSTEM[_norm(key)] = value
        return
    if scope == "domain":
        CANONICAL_TAU_BY_DOMAIN[_norm(key)] = value
        return
    raise ValueError("scope must be: system | domain")


def resolve_characteristic_time(*, system=None, domain=None, tau=None, default=None):
    """Resolve ``tau`` without a signal-derived or epsilon fallback."""
    if tau is not None and _norm(tau) not in {"auto", "canonical", "default"}:
        return _positive(tau)

    system_key = _norm(system)
    domain_key = _norm(domain)
    if system_key and system_key in CANONICAL_TAU_BY_SYSTEM:
        return CANONICAL_TAU_BY_SYSTEM[system_key]
    if domain_key and domain_key in CANONICAL_TAU_BY_DOMAIN:
        return CANONICAL_TAU_BY_DOMAIN[domain_key]
    if default is not None:
        return _positive(default)

    raise ValueError(
        "tau is a structural physical parameter; provide it explicitly or register a "
        "physical convention for the system/domain"
    )


def tau_context_from_metadata(metadata: Mapping[str, Any] | None):
    """Extract characteristic-time context from metadata."""
    if metadata is None:
        return {}
    return {
        "system": metadata.get("system_type"),
        "domain": metadata.get("domain"),
        "tau": metadata.get("characteristic_time"),
    }


__all__ = [
    "CANONICAL_TAU_BY_SYSTEM",
    "CANONICAL_TAU_BY_DOMAIN",
    "register_characteristic_time",
    "resolve_characteristic_time",
    "tau_context_from_metadata",
]
