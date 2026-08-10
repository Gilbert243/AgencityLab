"""Stable public orchestration for the canonical Agencity ``u -> b`` pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from agencitylab.backends.selector import select_backend
from agencitylab.config.runtime import get_runtime_config
from agencitylab.config.schema import validate_config
from agencitylab.core.activation import activation, reduced_coordinate
from agencitylab.core.activity import activity
from agencitylab.core.agencity import agencity
from agencitylab.core.beta import compute_beta
from agencitylab.core.intensity import compute_intensities
from agencitylab.core.memory import memory
from agencitylab.core.normalization import normalize_signal
from agencitylab.core.organization import organization
from agencitylab.core.power import characteristic_power
from agencitylab.core.tau import characteristic_time
from agencitylab.core.validation import is_exactly_constant, validate_positive_scalar
from agencitylab.exceptions import AgencityValidationError, PhysicalParameterError
from agencitylab.models import AgencityResult, ExperimentMetadata

from .presets import resolve_compute_config
from .validation import prepare_inputs, validate_physical_context

_CONFIG_OVERRIDE_KEYS = {"backend", "prefer_gpu", "normalization_method"}
_AUTO_VALUES = {"auto", "canonical", "default"}


def _is_auto(value) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _AUTO_VALUES
    )


def _physical_error(name: str, exc: Exception) -> PhysicalParameterError:
    return PhysicalParameterError(f"{name} resolution failed: {exc}")


def _power_profile(P_c, xi):
    """Return an explicitly supplied sampled power profile, or ``None``."""
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
    if not np.all(np.isfinite(profile)) or np.any(profile <= 0.0):
        raise PhysicalParameterError(
            "time-varying P_c must contain only strictly positive finite values"
        )
    return profile


def compute_agencity(
    data=None,
    u=None,
    xi=None,
    *,
    unit: Optional[str] = None,
    coordinate_unit: Optional[str] = None,
    power_unit: Optional[str] = None,
    observable_kind: Optional[str] = None,
    domain: Optional[str] = None,
    mechanism: Optional[str] = None,
    system_type: Optional[str] = None,
    environment: Optional[str] = None,
    geometry: Optional[str] = None,
    A_ref: float | str | None = None,
    tau: float | str | None = "auto",
    w: float | None = None,
    P_c: Any = "auto",
    activity_factor: float | str | None = "auto",
    resolution_scale: float | None = None,
    preset: str | Dict[str, Any] = "default",
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[dict | ExperimentMetadata] = None,
    verbose: bool = False,
    **overrides,
) -> AgencityResult:
    """Compute the scalar-signal Theory of Agencity pipeline.

    ``A_ref`` and ``tau`` are physical/contextual inputs. ``P_c`` may be a
    strictly positive scalar, a strictly positive sampled profile matching
    ``xi``, or a callable evaluated on ``xi``. A scalar may also be carried by
    metadata or a deliberately registered physical convention. No power profile
    is inferred from the observable signal.

    The CRM width ``w`` is a theory parameter distinct from ``tau`` in Volume 2.
    When ``w`` is omitted, AgencityLab uses the common convention ``w = tau``;
    an explicitly supplied positive ``w`` is preserved exactly and recorded in
    the result metadata. No signal statistic is used to choose ``w`` here.

    Unit arguments are descriptive labels only. ``unit`` labels ``u`` and
    ``A_ref``; ``coordinate_unit`` labels ``xi``, ``tau`` and ``w``;
    ``power_unit`` labels ``P_c``. The result reports ``b`` with the
    corresponding informational-power label (for example ``W·nat``).
    AgencityLab never silently converts magnitudes.

    ``data`` remains a compatibility alias for ``u``. Supplying both is an error.
    Unknown keyword arguments are rejected instead of being silently ignored.
    """
    if activity_factor not in {None, "auto"}:
        raise PhysicalParameterError(
            "activity_factor is legacy metadata and cannot alter the CRM"
        )
    if resolution_scale is not None:
        raise AgencityValidationError(
            "resolution_scale preprocessing is outside the canonical core; apply "
            "instrument preprocessing explicitly before compute_agencity"
        )

    if "Pc" in overrides:
        if not _is_auto(P_c):
            raise AgencityValidationError("provide only one of 'P_c' or legacy alias 'Pc'")
        P_c = overrides.pop("Pc")
    if "A_fact" in overrides:
        raise PhysicalParameterError(
            "A_fact/activity_factor no longer modifies the computation"
        )

    unknown = sorted(set(overrides) - _CONFIG_OVERRIDE_KEYS)
    if unknown:
        names = ", ".join(unknown)
        raise AgencityValidationError(f"unexpected compute_agencity keyword(s): {names}")

    if config is not None and not isinstance(config, dict):
        raise AgencityValidationError("config must be a dictionary or None")

    runtime_cfg = get_runtime_config()
    merged_config = runtime_cfg.to_dict() if runtime_cfg is not None else {}
    if config:
        merged_config.update(config)
    merged_config.update(overrides)

    try:
        cfg = validate_config(resolve_compute_config(preset, config=merged_config)).to_dict()
    except (KeyError, ValueError, TypeError) as exc:
        raise AgencityValidationError(f"invalid compute configuration: {exc}") from exc

    normalization_method = str(cfg.get("normalization_method", "A_ref")).strip().lower()
    if normalization_method not in {"a_ref", "canonical", "auto", "default"}:
        raise AgencityValidationError(
            "compute_agencity requires normalization_method='A_ref'"
        )

    backend_name = cfg.get("backend", "numpy")
    try:
        backend = select_backend(
            backend_name,
            auto=(backend_name == "auto"),
            prefer_gpu=bool(cfg.get("prefer_gpu", False)),
        )
    except Exception as exc:
        raise AgencityValidationError(f"backend selection failed: {exc}") from exc
    if verbose:
        print(f"[backend] {backend['name']}")

    xi_was_provided = xi is not None
    xi, u = prepare_inputs(data=data, u=u, xi=xi)

    meta_dict = validate_physical_context(
        metadata,
        unit=unit,
        coordinate_unit=coordinate_unit,
        power_unit=power_unit,
        observable_kind=observable_kind,
        domain=domain,
        reference_amplitude=None if _is_auto(A_ref) else A_ref,
    )
    meta_dict["mechanism"] = mechanism or meta_dict.get("mechanism", "")
    meta_dict["system_type"] = system_type or meta_dict.get("system_type", "")
    meta_dict["environment"] = environment or meta_dict.get("environment", "")
    meta_dict["geometry"] = geometry or meta_dict.get("geometry", "")
    if not meta_dict.get("coordinate_unit") and not xi_was_provided:
        meta_dict["coordinate_unit"] = "sample"

    try:
        meta = ExperimentMetadata.from_dict(meta_dict)
    except ValueError as exc:
        raise AgencityValidationError(f"invalid metadata: {exc}") from exc

    try:
        u_star, A_ref_used = normalize_signal(
            u,
            A_ref=None if _is_auto(A_ref) else A_ref,
            unit=meta.unit,
            observable_kind=meta.observable_kind,
            domain=meta.domain,
            metadata=meta.to_dict(),
            method="canonical",
            verbose=verbose,
        )
    except ValueError as exc:
        raise _physical_error("A_ref", exc) from exc
    meta.reference_amplitude = float(A_ref_used)

    try:
        tau_eff = characteristic_time(
            tau=tau,
            metadata=meta.to_dict(),
            domain=meta.domain,
            system=meta.system_type,
            verbose=verbose,
        )
    except ValueError as exc:
        raise _physical_error("tau", exc) from exc
    meta.characteristic_time = tau_eff

    if w is None:
        memory_window = tau_eff
    else:
        try:
            memory_window = validate_positive_scalar(w, name="w")
        except ValueError as exc:
            raise PhysicalParameterError(str(exc)) from exc
    meta.memory_window = memory_window
    # Keep the historical metadata.extra read path while exposing the typed field.
    meta.extra["memory_window"] = memory_window
    meta.extra["memory_window_mode"] = "w=tau default" if w is None else "explicit"

    profile = _power_profile(P_c, xi)
    if profile is not None:
        P_eff = profile
        meta.characteristic_power = None
        meta.extra["characteristic_power_mode"] = "time_varying"
    else:
        try:
            P_eff = characteristic_power(
                value=None if _is_auto(P_c) else P_c,
                system=meta.system_type,
                domain=meta.domain,
                tau=tau_eff,
                metadata=meta.to_dict(),
                verbose=verbose,
            )
        except ValueError as exc:
            raise _physical_error("P_c", exc) from exc
        meta.characteristic_power = float(P_eff)
        meta.extra.pop("characteristic_power_mode", None)

    t_star = reduced_coordinate(xi, tau_eff)

    if is_exactly_constant(u):
        if verbose:
            print("[canonical] exact rest state detected; derivative/CRM stages bypassed")
        zeros = np.zeros_like(u_star, dtype=float)
        complex_zeros = np.zeros_like(u_star, dtype=complex)
        X_star = zeros.copy()
        A_star = zeros.copy()
        M = zeros.copy()
        O = zeros.copy()
        D = zeros.copy()
        S = zeros.copy()
        J = zeros.copy()
        U = complex_zeros.copy()
        beta = complex_zeros.copy()
        b = complex_zeros.copy()
    else:
        try:
            X_star = activation(u_star, axis=t_star, verbose=verbose)
            A_star = activity(X_star, axis=t_star, verbose=verbose)
            M = memory(
                u_star,
                tau_eff,
                axis=xi,
                window=memory_window,
                verbose=verbose,
            )
            O = organization(
                u_star,
                X_star,
                tau_eff,
                axis=xi,
                window=memory_window,
                verbose=verbose,
            )
            D, S = compute_intensities(X_star, A_star, M, O, verbose=verbose)
            J, U, beta = compute_beta(D, S, M, O, verbose=verbose)
            b = agencity(beta, P_eff, verbose=verbose)
        except ValueError as exc:
            raise AgencityValidationError(f"numerical pipeline failed: {exc}") from exc

    return AgencityResult(
        xi=xi,
        u=u,
        u_star=u_star,
        X_star=X_star,
        A_star=A_star,
        tau=tau_eff,
        t_star=t_star,
        M=M,
        O=O,
        D=D,
        S=S,
        J=J,
        U=U,
        beta=beta,
        b_reduced=beta,
        b=b,
        P_c=P_eff,
        A_ref=float(meta.reference_amplitude),
        A_fact=1.0,
        resolution_scale=None,
        unit=meta.unit,
        coordinate_unit=meta.coordinate_unit,
        power_unit=meta.power_unit,
        observable_kind=meta.observable_kind,
        domain=meta.domain,
        system_type=meta.system_type,
        mechanism=meta.mechanism,
        metadata=meta,
        config=dict(cfg),
    )
