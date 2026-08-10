"""
High-level validation helpers for the AgencityLab public API.

This layer validates user inputs before delegating to the core engine.
It stays lightweight and does not duplicate the core mathematics.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from agencitylab.core.validation import (
    as_float_array,
    validate_axis,
    validate_signal,
    validate_window_size,
)
from agencitylab.core.safeguards import ensure_positive


def prepare_inputs(
    data=None,
    u=None,
    xi=None,
    *,
    name: str = "signal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare and validate (xi, u).
    """
    signal = u if u is not None else data
    if signal is None:
        raise ValueError("Either 'u' or 'data' must be provided")

    signal = validate_signal(signal, name=name)
    signal = np.asarray(signal, dtype=float)

    if xi is None:
        xi = np.arange(signal.shape[0], dtype=float)
    else:
        xi = validate_axis(xi, expected_length=signal.shape[0], name="xi")

    return xi, signal


def validate_optional_tau(tau: Optional[float]):
    """Validate an optional tau parameter."""
    if tau is None:
        return None
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive")
    return tau


def validate_optional_power(P_c):
    """
    Validate an optional characteristic power input.
    """
    if P_c is None:
        return None

    if callable(P_c):
        return P_c

    arr = np.asarray(P_c)
    if arr.ndim == 0:
        return ensure_positive(float(arr))

    if not np.all(np.isfinite(arr)):
        raise ValueError("P_c must contain only finite values")

    return ensure_positive(arr)


def validate_kind(kind: str, allowed: Iterable[str]):
    """Validate a visualization or analysis kind."""
    kind = str(kind).lower().strip()
    allowed = {str(x).lower().strip() for x in allowed}
    if kind not in allowed:
        raise ValueError(f"Unknown kind '{kind}'. Allowed: {sorted(allowed)}")
    return kind


def validate_metadata(metadata: Optional[Dict[str, Any]]):
    """
    Validate metadata dictionary.

    This function preserves all keys so that model-level metadata can carry
    physics context such as unit, observable_kind, domain and reference_amplitude.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary or None")
    return dict(metadata)


def validate_physical_context(
    metadata: Optional[Dict[str, Any]] = None,
    *,
    unit: Optional[str] = None,
    observable_kind: Optional[str] = None,
    domain: Optional[str] = None,
    reference_amplitude: Any = None,
):
    """
    Merge and normalize the physical context used to resolve A_ref.
    """
    meta = validate_metadata(metadata)

    if unit is not None:
        meta["unit"] = unit
    if observable_kind is not None:
        meta["observable_kind"] = observable_kind
    if domain is not None:
        meta["domain"] = domain
    if reference_amplitude is not None:
        meta["reference_amplitude"] = reference_amplitude

    return meta


def validate_batch_items(items):
    """Validate batch input container."""
    if items is None:
        raise ValueError("Batch items cannot be None")
    try:
        items = list(items)
    except TypeError as exc:
        raise ValueError("Batch items must be iterable") from exc
    if len(items) == 0:
        raise ValueError("Batch items cannot be empty")
    return items