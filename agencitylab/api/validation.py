"""Validation helpers for the stable AgencityLab public API.

This layer validates user-facing contracts before delegating mathematical work
to :mod:`agencitylab.core`. It does not redefine canonical equations.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from agencitylab.exceptions import AgencityValidationError, UnitValidationError
from agencitylab.models.metadata import ExperimentMetadata


def normalize_unit_label(value: Optional[str], *, name: str) -> str:
    """Normalize an optional descriptive unit label without converting values."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise UnitValidationError(f"{name} must be a string or None")
    return value.strip()


def prepare_inputs(
    data=None,
    u=None,
    xi=None,
    *,
    name: str = "u",
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare one finite one-dimensional observable and coordinate array."""
    if u is not None and data is not None:
        raise AgencityValidationError("provide only one of 'u' or its compatibility alias 'data'")

    signal = u if u is not None else data
    if signal is None:
        raise AgencityValidationError("either 'u' or 'data' must be provided")

    try:
        signal = np.asarray(signal, dtype=float)
    except Exception as exc:
        raise AgencityValidationError(f"{name} must be numeric") from exc

    if signal.ndim != 1:
        raise AgencityValidationError(
            f"{name} must be one-dimensional for the canonical scalar API"
        )
    if signal.size < 3:
        raise AgencityValidationError(f"{name} must contain at least three samples")
    if not np.all(np.isfinite(signal)):
        raise AgencityValidationError(f"{name} must contain only finite values")

    if xi is None:
        axis = np.arange(signal.size, dtype=float)
    else:
        try:
            axis = np.asarray(xi, dtype=float)
        except Exception as exc:
            raise AgencityValidationError("xi must be numeric") from exc
        if axis.ndim != 1:
            raise AgencityValidationError("xi must be one-dimensional")
        if axis.size != signal.size:
            raise AgencityValidationError("xi and u must have the same length")
        if not np.all(np.isfinite(axis)):
            raise AgencityValidationError("xi must contain only finite values")
        if np.any(np.diff(axis) <= 0.0):
            raise AgencityValidationError("xi must be strictly increasing")

    return axis, signal


def validate_optional_tau(tau: Optional[float]):
    """Validate an optional positive structural time without epsilon substitution."""
    if tau is None:
        return None
    try:
        value = float(tau)
    except Exception as exc:
        raise AgencityValidationError("tau must be numeric") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise AgencityValidationError("tau must be strictly positive")
    return value


def validate_optional_power(P_c):
    """Validate an optional characteristic-power value for compatibility helpers."""
    if P_c is None or callable(P_c):
        return P_c
    arr = np.asarray(P_c, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise AgencityValidationError("P_c must contain only strictly positive finite values")
    return float(arr) if arr.ndim == 0 else arr


def validate_kind(kind: str, allowed: Iterable[str]):
    """Validate a visualization or analysis kind."""
    key = str(kind).lower().strip()
    allowed_values = {str(value).lower().strip() for value in allowed}
    if key not in allowed_values:
        raise AgencityValidationError(
            f"unknown kind '{kind}'; allowed values are {sorted(allowed_values)}"
        )
    return key


def validate_metadata(
    metadata: Optional[Dict[str, Any] | ExperimentMetadata],
) -> Dict[str, Any]:
    """Return validated metadata as a detached dictionary."""
    try:
        return ExperimentMetadata.from_dict(metadata).to_dict()
    except ValueError as exc:
        raise AgencityValidationError(str(exc)) from exc


def validate_physical_context(
    metadata: Optional[Dict[str, Any] | ExperimentMetadata] = None,
    *,
    unit: Optional[str] = None,
    coordinate_unit: Optional[str] = None,
    power_unit: Optional[str] = None,
    observable_kind: Optional[str] = None,
    domain: Optional[str] = None,
    reference_amplitude: Any = None,
) -> Dict[str, Any]:
    """Merge explicit public arguments into reproducibility metadata."""
    meta = validate_metadata(metadata)

    if unit is not None:
        meta["unit"] = normalize_unit_label(unit, name="unit")
    if coordinate_unit is not None:
        meta["coordinate_unit"] = normalize_unit_label(
            coordinate_unit, name="coordinate_unit"
        )
    if power_unit is not None:
        meta["power_unit"] = normalize_unit_label(power_unit, name="power_unit")
    if observable_kind is not None:
        meta["observable_kind"] = str(observable_kind).strip()
    if domain is not None:
        meta["domain"] = str(domain).strip()
    if reference_amplitude is not None:
        meta["reference_amplitude"] = reference_amplitude

    try:
        return ExperimentMetadata.from_dict(meta).to_dict()
    except ValueError as exc:
        raise AgencityValidationError(str(exc)) from exc


def validate_batch_items(items):
    """Validate and materialize a non-empty batch iterable."""
    if items is None:
        raise AgencityValidationError("batch items cannot be None")
    try:
        materialized = list(items)
    except TypeError as exc:
        raise AgencityValidationError("batch items must be iterable") from exc
    if not materialized:
        raise AgencityValidationError("batch items cannot be empty")
    return materialized
