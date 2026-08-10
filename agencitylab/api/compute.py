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
from agencitylab.models import AgencityResult, ExperimentMetadata

from .presets import resolve_compute_config
from .validation import prepare_inputs, validate_physical_context


def _is_auto(value) -> bool:
    return value is None or (
        isinstance(value, str)
        and value.strip().lower() in {"auto", "canonical", "default"}
    )


def compute_agencity(
    data=None,
    u=None,
    xi=None,
    *,
    unit: Optional[str] = None,
    observable_kind: Optional[str] = None,
    domain: Optional[str] = None,
    mechanism: Optional[str] = None,
    system_type: Optional[str] = None,
    environment: Optional[str] = None,
    geometry: Optional[str] = None,
    A_ref: float | str | None = None,
    tau: float | str | None = "auto",
    w: float | None = None,
    P_c: float | str | None = "auto",
    activity_factor: float | str | None = "auto",
    resolution_scale: float | None = None,
    preset: str | Dict[str, Any] = "default",
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[dict] = None,
    verbose: bool = False,
    **overrides,
) -> AgencityResult:
    """Compute the canonical Theory of Agencity pipeline.

    Physical/contextual inputs ``A_ref``, ``tau`` and ``P_c`` must be explicit,
    present in metadata, or resolved from a deliberately registered convention.
    The canonical CRM window is exactly ``w = tau``. No signal-derived physical
    parameter, saturation, smoothing, or epsilon-modified equation is used.

    The exact sampled rest state is treated as a canonical null-state postulate:
    once ``u`` is verified to be exactly constant, the pipeline does not attempt to
    establish nullity through numerical derivatives or CRM. It returns
    ``X*=A*=M=O=D=S=J=U=beta=b=0`` exactly.
    """
    if activity_factor not in {None, "auto"}:
        raise ValueError("activity_factor is legacy and cannot alter the canonical CRM")
    if resolution_scale is not None:
        raise ValueError("resolution_scale preprocessing is outside the canonical core")

    runtime_cfg = get_runtime_config()
    merged_config = runtime_cfg.to_dict() if runtime_cfg is not None else {}
    if config is not None:
        merged_config.update(config)
    cfg = validate_config(
        resolve_compute_config(preset, config=merged_config, overrides=overrides)
    ).to_dict()

    normalization_method = str(cfg.get("normalization_method", "A_ref")).strip().lower()
    if normalization_method not in {"a_ref", "canonical", "auto", "default"}:
        raise ValueError("compute_agencity canonical mode requires A_ref normalization")

    backend_name = cfg.get("backend", "numpy")
    backend = select_backend(
        backend_name,
        auto=(backend_name == "auto"),
        prefer_gpu=bool(cfg.get("prefer_gpu", False)),
    )
    if verbose:
        print(f"[backend] {backend['name']}")

    xi, u = prepare_inputs(data=data, u=u, xi=xi)
    meta_dict = validate_physical_context(
        metadata,
        unit=unit,
        observable_kind=observable_kind,
        domain=domain,
        reference_amplitude=A_ref,
    )
    meta_dict["mechanism"] = mechanism or meta_dict.get("mechanism", "")
    meta_dict["system_type"] = system_type or meta_dict.get("system_type", "")
    meta_dict["environment"] = environment or meta_dict.get("environment", "")
    meta_dict["geometry"] = geometry or meta_dict.get("geometry", "")
    meta = ExperimentMetadata.from_dict(meta_dict)

    u_star, A_ref_used = normalize_signal(
        u,
        A_ref=A_ref,
        unit=meta.unit,
        observable_kind=meta.observable_kind,
        domain=meta.domain,
        metadata=meta.to_dict(),
        method="canonical",
        verbose=verbose,
    )
    meta.reference_amplitude = float(A_ref_used)

    tau_eff = characteristic_time(
        tau=tau,
        metadata=meta.to_dict(),
        domain=meta.domain,
        system=meta.system_type,
        verbose=verbose,
    )
    meta.characteristic_time = tau_eff

    if w is None:
        memory_window = tau_eff
    else:
        memory_window = validate_positive_scalar(w, name="w")
        if memory_window != tau_eff:
            raise ValueError("canonical CRM requires w = tau exactly")
    meta.extra["memory_window"] = memory_window

    P_eff = characteristic_power(
        value=None if _is_auto(P_c) else P_c,
        system=meta.system_type,
        domain=meta.domain,
        tau=tau_eff,
        metadata=meta.to_dict(),
        verbose=verbose,
    )
    meta.characteristic_power = float(P_eff)

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
        observable_kind=meta.observable_kind,
        domain=meta.domain,
        system_type=meta.system_type,
        mechanism=meta.mechanism,
        metadata=meta,
        config=dict(cfg),
    )
