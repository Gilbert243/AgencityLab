"""Physical reference amplitudes used by canonical normalization.

Only explicitly documented conventions are pre-registered. Unknown observables must
provide ``A_ref`` explicitly or register a stable domain convention; the canonical core
never invents a scale from the measured signal.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CANONICAL_AREF_BY_KIND: dict[str, float] = {
    "dimensionless": 1.0,
    "temperature": 1.0,
    "angle": 1.0,
    "small_angle": 0.1,
    "voltage": 1.0,
}

CANONICAL_AREF_BY_UNIT: dict[str, float] = {
    "1": 1.0,
    "dimensionless": 1.0,
    "k": 1.0,
    "rad": 1.0,
    "v": 1.0,
}

CANONICAL_AREF_BY_DOMAIN_KIND: dict[tuple[str, str], float] = {
    ("mechanics", "small_angle"): 0.1,
    ("angular", "small_angle"): 0.1,
}


def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _positive(value, *, name="A_ref") -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return out


def register_reference_amplitude(key, value, *, scope: str = "kind") -> None:
    """Register an explicit, reusable physical reference amplitude convention."""
    scope = _norm(scope)
    value = _positive(value)
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
    unit=None,
    observable_kind=None,
    domain=None,
    A_ref=None,
    default=None,
) -> float:
    """Resolve ``A_ref`` without signal-derived or epsilon-based fallback."""
    if A_ref is not None and _norm(A_ref) not in {"auto", "canonical", "default"}:
        return _positive(A_ref)

    domain_key = _norm(domain)
    kind_key = _norm(observable_kind)
    unit_key = _norm(unit)

    if domain_key and kind_key:
        value = CANONICAL_AREF_BY_DOMAIN_KIND.get((domain_key, kind_key))
        if value is not None:
            return value
    if kind_key and kind_key in CANONICAL_AREF_BY_KIND:
        return CANONICAL_AREF_BY_KIND[kind_key]
    if unit_key and unit_key in CANONICAL_AREF_BY_UNIT:
        return CANONICAL_AREF_BY_UNIT[unit_key]
    if default is not None:
        return _positive(default)

    raise ValueError(
        "A_ref is required for this observable; provide a physical reference amplitude "
        "or register a stable convention"
    )


def resolve_reference_amplitudes(
    *, units=None, observable_kinds=None, domains=None, A_refs=None, default=None
) -> np.ndarray:
    """Vectorized reference-amplitude resolution."""
    if not isinstance(units, (list, tuple, np.ndarray)):
        units = [units]
    n = len(units)
    if not isinstance(observable_kinds, (list, tuple, np.ndarray)):
        observable_kinds = [observable_kinds] * n
    if not isinstance(domains, (list, tuple, np.ndarray)):
        domains = [domains] * n
    if not isinstance(A_refs, (list, tuple, np.ndarray)):
        A_refs = [A_refs] * n
    return np.asarray(
        [
            resolve_reference_amplitude(
                unit=units[i],
                observable_kind=observable_kinds[i],
                domain=domains[i],
                A_ref=A_refs[i],
                default=default,
            )
            for i in range(n)
        ],
        dtype=float,
    )


def reference_context_from_metadata(metadata: Mapping[str, Any] | None):
    """Extract physical reference-amplitude context from metadata."""
    if metadata is None:
        return {}
    return {
        "unit": metadata.get("unit"),
        "observable_kind": metadata.get("observable_kind"),
        "domain": metadata.get("domain"),
        "A_ref": metadata.get("reference_amplitude"),
    }


__all__ = [
    "CANONICAL_AREF_BY_KIND",
    "CANONICAL_AREF_BY_UNIT",
    "CANONICAL_AREF_BY_DOMAIN_KIND",
    "register_reference_amplitude",
    "resolve_reference_amplitude",
    "resolve_reference_amplitudes",
    "reference_context_from_metadata",
]
