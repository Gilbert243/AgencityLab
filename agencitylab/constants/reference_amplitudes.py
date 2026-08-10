"""
Canonical reference amplitudes for AgencityLab.

A_ref is a fixed property of the observable and its physical domain.
It is NOT estimated from the signal.

Canonical definition:
    u*(t) = u(t) / A_ref

The registry below defines domain conventions.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

# Safe fallback
try:
    from agencitylab.core.safeguards import ensure_positive
except Exception:
    def ensure_positive(x, minimum=1e-12):
        return max(float(x), minimum)


# ============================================================
# CANONICAL REGISTRIES
# ============================================================

CANONICAL_AREF_BY_KIND: dict[str, float] = {
    "dimensionless": 1.0,
    "position": 1.0,
    "length": 1.0,
    "temperature": 1.0,
    "angle": 1.0,
    "small_angle": 0.1,
    "voltage": 1.0,
    "current": 1.0,
    "time": 1.0,
    "frequency": 1.0,
    "speed": 1.0,
    "acceleration": 1.0,
    "force": 1.0,
    "power": 1.0,
    "pressure": 1.0,
}

CANONICAL_AREF_BY_UNIT: dict[str, float] = {
    "1": 1.0,
    "dimensionless": 1.0,
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "cm": 1.0,
    "mm": 1.0,
    "km": 1.0,
    "rad": 1.0,
    "deg": 1.0,
    "degree": 1.0,
    "°": 1.0,
    "°c": 1.0,
    "c": 1.0,
    "k": 1.0,
    "v": 1.0,
    "a": 1.0,
    "s": 1.0,
    "hz": 1.0,
    "n": 1.0,
    "pa": 1.0,
    "w": 1.0,
}

CANONICAL_AREF_BY_DOMAIN_KIND: dict[tuple[str, str], float] = {
    ("mechanics", "small_angle"): 0.1,
    ("angular", "small_angle"): 0.1,
}


# ============================================================
# INTERNAL UTILS
# ============================================================

def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


# ============================================================
# PUBLIC API
# ============================================================

def register_reference_amplitude(
    key: str | tuple[str, str],
    value: float,
    *,
    scope: str = "kind",
) -> None:
    """
    Register or override a canonical A_ref.

    scope:
        - "kind"
        - "unit"
        - "domain_kind"
    """
    scope = _norm(scope)
    value = float(ensure_positive(value))

    if scope == "kind":
        CANONICAL_AREF_BY_KIND[_norm(key)] = value
        return

    if scope == "unit":
        CANONICAL_AREF_BY_UNIT[_norm(key)] = value
        return

    if scope == "domain_kind":
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError("Expected (domain, kind)")
        domain, kind = key
        CANONICAL_AREF_BY_DOMAIN_KIND[(_norm(domain), _norm(kind))] = value
        return

    raise ValueError("scope must be: kind | unit | domain_kind")


def resolve_reference_amplitude(
    *,
    unit: Any = None,
    observable_kind: Any = None,
    domain: Any = None,
    A_ref: Any = None,
    default: float = 1.0,
) -> float:
    """
    Resolve A_ref according to theory.

    Priority:
        1) explicit override (if not 'auto')
        2) (domain, kind)
        3) kind
        4) unit
        5) default
    """
    # Explicit override
    if A_ref is not None:
        token = _norm(A_ref)
        if token not in {"auto", "canonical", "default"}:
            return float(ensure_positive(A_ref))

    domain_key = _norm(domain)
    kind_key = _norm(observable_kind)
    unit_key = _norm(unit)

    # Domain + kind
    if domain_key and kind_key:
        val = CANONICAL_AREF_BY_DOMAIN_KIND.get((domain_key, kind_key))
        if val is not None:
            return val

    # Kind
    if kind_key:
        val = CANONICAL_AREF_BY_KIND.get(kind_key)
        if val is not None:
            return val

    # Unit
    if unit_key:
        val = CANONICAL_AREF_BY_UNIT.get(unit_key)
        if val is not None:
            return val

    return float(ensure_positive(default))


def resolve_reference_amplitudes(
    *,
    units=None,
    observable_kinds=None,
    domains=None,
    A_refs=None,
    default: float = 1.0,
) -> np.ndarray:
    """
    Vectorized version for multivariate signals.
    """
    if not isinstance(units, (list, tuple, np.ndarray)):
        units = [units]

    n = len(units)

    if not isinstance(observable_kinds, (list, tuple, np.ndarray)):
        observable_kinds = [observable_kinds] * n

    if not isinstance(domains, (list, tuple, np.ndarray)):
        domains = [domains] * n

    if not isinstance(A_refs, (list, tuple, np.ndarray)):
        A_refs = [A_refs] * n

    out = np.empty(n, dtype=float)

    for i in range(n):
        out[i] = resolve_reference_amplitude(
            unit=units[i],
            observable_kind=observable_kinds[i],
            domain=domains[i],
            A_ref=A_refs[i],
            default=default,
        )

    return out


def reference_context_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Extract A_ref context from metadata.
    """
    if metadata is None:
        return {}

    return {
        "unit": metadata.get("unit"),
        "observable_kind": metadata.get("observable_kind"),
        "domain": metadata.get("domain"),
        "A_ref": metadata.get("reference_amplitude"),
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "CANONICAL_AREF_BY_KIND",
    "CANONICAL_AREF_BY_UNIT",
    "CANONICAL_AREF_BY_DOMAIN_KIND",
    "register_reference_amplitude",
    "resolve_reference_amplitude",
    "resolve_reference_amplitudes",
    "reference_context_from_metadata",
]