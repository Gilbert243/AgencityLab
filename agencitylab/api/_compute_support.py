"""Private context-resolution helpers for :func:`compute_agencity`.

These helpers resolve user input, physical/contextual metadata and software
bookkeeping. They do not define or alter any canonical Theory of Agencity
equation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from agencitylab.core.normalization import normalize_signal
from agencitylab.core.power import characteristic_power
from agencitylab.core.tau import characteristic_time
from agencitylab.core.validation import validate_positive_scalar
from agencitylab.exceptions import AgencityValidationError, PhysicalParameterError
from agencitylab.models import ExperimentMetadata
from agencitylab.version import __version__

from .validation import validate_physical_context

FloatArray: TypeAlias = NDArray[np.float64]
PowerCallable: TypeAlias = Callable[[FloatArray], ArrayLike]
PowerInput: TypeAlias = float | ArrayLike | PowerCallable | str | None
MetadataInput: TypeAlias = Mapping[str, Any] | ExperimentMetadata | None

_AUTO_VALUES = {"auto", "canonical", "default"}


def is_auto(value: Any) -> bool:
    """Whether a public physical input requests registered contextual resolution."""
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _AUTO_VALUES
    )


def _physical_error(name: str, exc: Exception) -> PhysicalParameterError:
    return PhysicalParameterError(f"{name} resolution failed: {exc}")


def prepare_metadata(
    metadata: MetadataInput,
    *,
    xi_was_provided: bool,
    unit: str | None,
    coordinate_unit: str | None,
    power_unit: str | None,
    observable_kind: str | None,
    domain: str | None,
    mechanism: str | None,
    system_type: str | None,
    environment: str | None,
    geometry: str | None,
    A_ref: float | str | None,
) -> ExperimentMetadata:
    """Build validated reproducibility metadata from explicit public inputs."""
    meta_dict = validate_physical_context(
        metadata,
        unit=unit,
        coordinate_unit=coordinate_unit,
        power_unit=power_unit,
        observable_kind=observable_kind,
        domain=domain,
        reference_amplitude=None if is_auto(A_ref) else A_ref,
    )
    meta_dict["mechanism"] = mechanism or meta_dict.get("mechanism", "")
    meta_dict["system_type"] = system_type or meta_dict.get("system_type", "")
    meta_dict["environment"] = environment or meta_dict.get("environment", "")
    meta_dict["geometry"] = geometry or meta_dict.get("geometry", "")
    if not meta_dict.get("coordinate_unit") and not xi_was_provided:
        meta_dict["coordinate_unit"] = "sample"

    try:
        metadata_model = ExperimentMetadata.from_dict(meta_dict)
    except ValueError as exc:
        raise AgencityValidationError(f"invalid metadata: {exc}") from exc
    metadata_model.agencitylab_version = __version__
    return metadata_model


def resolve_normalized_observable(
    u: FloatArray,
    *,
    A_ref: float | str | None,
    metadata: ExperimentMetadata,
    verbose: bool,
) -> tuple[FloatArray, float]:
    """Resolve ``A_ref`` and return canonical ``u*`` without heuristic scaling."""
    try:
        u_star, A_ref_used = normalize_signal(
            u,
            A_ref=None if is_auto(A_ref) else A_ref,
            unit=metadata.unit,
            observable_kind=metadata.observable_kind,
            domain=metadata.domain,
            metadata=metadata.to_dict(),
            method="canonical",
            verbose=verbose,
        )
    except ValueError as exc:
        raise _physical_error("A_ref", exc) from exc
    metadata.reference_amplitude = float(A_ref_used)
    return np.asarray(u_star, dtype=float), float(A_ref_used)


def resolve_characteristic_time(
    tau: float | str | None,
    *,
    metadata: ExperimentMetadata,
    verbose: bool,
) -> float:
    """Resolve the explicit/contextual structural time ``tau``."""
    try:
        tau_eff = characteristic_time(
            tau=tau,
            metadata=metadata.to_dict(),
            domain=metadata.domain,
            system=metadata.system_type,
            verbose=verbose,
        )
    except ValueError as exc:
        raise _physical_error("tau", exc) from exc
    metadata.characteristic_time = tau_eff
    return float(tau_eff)


def resolve_memory_window(
    w: float | None,
    *,
    tau: float,
    metadata: ExperimentMetadata,
) -> float:
    """Resolve the CRM width while keeping the ``w = tau`` fallback explicit."""
    if w is None:
        memory_window = tau
        mode = "w=tau default"
        convention = "w was unspecified; implementation convention w = tau was used"
    else:
        try:
            memory_window = validate_positive_scalar(w, name="w")
        except ValueError as exc:
            raise PhysicalParameterError(str(exc)) from exc
        mode = "explicit"
        convention = "w was supplied explicitly and preserved"

    metadata.memory_window = float(memory_window)
    metadata.extra["memory_window"] = float(memory_window)
    metadata.extra["memory_window_mode"] = mode
    metadata.extra["memory_window_convention"] = convention
    return float(memory_window)


def _power_profile(P_c: PowerInput, xi: FloatArray) -> FloatArray | None:
    if callable(P_c):
        try:
            candidate = P_c(xi)
        except Exception as exc:
            raise PhysicalParameterError(f"P_c callable failed: {exc}") from exc
    elif P_c is not None and not isinstance(P_c, str):
        candidate = P_c
        try:
            probe = np.asarray(candidate)
        except Exception as exc:
            raise PhysicalParameterError("P_c must be numeric") from exc
        if probe.ndim == 0:
            return None
    else:
        return None

    try:
        profile = np.asarray(candidate, dtype=float)
    except Exception as exc:
        raise PhysicalParameterError("time-varying P_c must be numeric") from exc
    if profile.ndim != 1 or profile.shape != xi.shape:
        raise PhysicalParameterError(
            "time-varying P_c must be one-dimensional and have the same length as xi"
        )
    if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
        raise PhysicalParameterError(
            "time-varying P_c must contain only non-negative finite values"
        )
    return profile


def resolve_characteristic_power(
    P_c: PowerInput,
    *,
    xi: FloatArray,
    tau: float,
    metadata: ExperimentMetadata,
    verbose: bool,
) -> float | FloatArray:
    """Resolve scalar ``P_c`` or preserve an explicitly supplied sampled profile."""
    profile = _power_profile(P_c, xi)
    if profile is not None:
        metadata.characteristic_power = None
        metadata.extra["characteristic_power_mode"] = "time_varying"
        return profile

    try:
        power = characteristic_power(
            value=None if is_auto(P_c) else P_c,
            system=metadata.system_type,
            domain=metadata.domain,
            tau=tau,
            metadata=metadata.to_dict(),
            verbose=verbose,
        )
    except ValueError as exc:
        raise _physical_error("P_c", exc) from exc

    metadata.characteristic_power = float(power)
    metadata.extra.pop("characteristic_power_mode", None)
    return float(power)
