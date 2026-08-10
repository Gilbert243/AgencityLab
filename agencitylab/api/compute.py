"""
compute.py

Canonical computation entry points for AgencityLab.

This module:
    - prepares inputs,
    - resolves structural parameters,
    - executes the canonical pipeline,
    - returns a structured AgencityResult.

Canonical pipeline:
    u
    → u*
    → X*
    → A*
    → M
    → O
    → D,S
    → J
    → U
    → beta
    → b
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
)

import numpy as np

# ============================================================
# BACKENDS
# ============================================================

from agencitylab.backends.selector import (
    select_backend,
)

# ============================================================
# CONFIG
# ============================================================

from agencitylab.config.runtime import (
    get_runtime_config,
)

from agencitylab.config.schema import (
    validate_config,
)

# ============================================================
# CORE
# ============================================================

from agencitylab.core.normalization import (
    normalize_signal,
)

from agencitylab.core.activation import (
    activation,
)

from agencitylab.core.activity import (
    activity,
)

from agencitylab.core.tau import (
    characteristic_time,
)

from agencitylab.core.memory import (
    memory,
)

from agencitylab.core.organization import (
    organization,
)

from agencitylab.core.intensity import (
    compute_intensities,
)

from agencitylab.core.beta import (
    compute_beta,
)

from agencitylab.core.agencity import (
    agencity,
)

from agencitylab.core.power import (
    characteristic_power,
)

# ============================================================
# MODELS
# ============================================================

from agencitylab.models import (
    AgencityResult,
    ExperimentMetadata,
)

# ============================================================
# API HELPERS
# ============================================================

from .validation import (
    prepare_inputs,
    validate_optional_tau,
    validate_metadata,
    validate_physical_context,
)

from .presets import (
    resolve_compute_config,
)

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _evaluate_power(
    P_c,
    xi,
):
    """
    Evaluate dynamic characteristic power.
    """

    if P_c is None:
        return None

    if callable(P_c):

        return np.asarray(
            P_c(xi)
        )

    arr = np.asarray(P_c)

    if arr.ndim == 0:
        return float(arr)

    return arr.astype(
        float,
        copy=False,
    )


# ============================================================
# MAIN API
# ============================================================

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
    P_c: float | Any | None = "auto",
    activity_factor: float | str | None = "auto",
    resolution_scale: float | None = None,
    preset: str | Dict[str, Any] = "default",
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[dict] = None,
    verbose: bool = False,
    **overrides,
) -> AgencityResult:
    """
    Compute canonical Agencity pipeline.
    """

    # ========================================================
    # CONFIG
    # ========================================================

    runtime_cfg = get_runtime_config()

    merged_config = {}

    if runtime_cfg is not None:

        merged_config.update(
            runtime_cfg.to_dict()
        )

    if config is not None:

        merged_config.update(config)

    cfg = resolve_compute_config(
        preset,
        config=merged_config,
        overrides=overrides,
    )

    cfg = validate_config(
        cfg
    ).to_dict()

    # ========================================================
    # BACKEND
    # ========================================================

    backend_name = cfg.get(
        "backend",
        "numpy",
    )

    prefer_gpu = bool(
        cfg.get(
            "prefer_gpu",
            False,
        )
    )

    backend = select_backend(
        backend_name,
        auto=(backend_name == "auto"),
        prefer_gpu=prefer_gpu,
    )

    if verbose:

        print(
            "[backend] "
            f"{backend['name']} "
            f"(gpu={prefer_gpu})"
        )

    # ========================================================
    # INPUTS
    # ========================================================

    xi, u = prepare_inputs(
        data=data,
        u=u,
        xi=xi,
    )

    if verbose:

        print(
            "[compute] "
            "Inputs prepared"
        )

        print(
            "[compute] "
            f"n_samples={len(u)}"
        )

    # ========================================================
    # METADATA
    # ========================================================

    meta_dict = validate_physical_context(
        metadata,
        unit=unit,
        observable_kind=observable_kind,
        domain=domain,
        reference_amplitude=A_ref,
    )

    # ========================================================
    # STRUCTURAL PARAMETERS
    # ========================================================

    meta_dict["mechanism"] = (
        mechanism
        or meta_dict.get(
            "mechanism",
            "",
        )
    )

    meta_dict["system_type"] = (
        system_type
        or meta_dict.get(
            "system_type",
            "",
        )
    )

    meta_dict["environment"] = (
        environment
        or meta_dict.get(
            "environment",
            "",
        )
    )

    meta_dict["geometry"] = (
        geometry
        or meta_dict.get(
            "geometry",
            "",
        )
    )

    meta_dict["activity_factor"] = (
        activity_factor
        if activity_factor != "auto"
        else meta_dict.get(
            "activity_factor",
            None,
        )
    )

    # ========================================================
    # RESOLUTION SCALE
    # ========================================================

    if resolution_scale is not None:

        meta_dict["resolution_scale"] = (
            float(resolution_scale)
        )

    meta = ExperimentMetadata.from_dict(
        meta_dict
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    u_star, A_ref_used = normalize_signal(
        u,
        A_ref=A_ref,
        unit=meta.unit,
        observable_kind=meta.observable_kind,
        domain=meta.domain,
        metadata=meta.to_dict(),
        method=str(
            cfg.get(
                "normalization_method",
                "A_ref",
            )
        ),
        center=False,
        axis=0,
        verbose=verbose,
    )

    meta.reference_amplitude = float(
        np.asarray(A_ref_used)
        .reshape(-1)[0]
    )

    if verbose:

        print(
            "[compute] "
            "Normalization complete"
        )

        print(
            "[compute] "
            f"A_ref="
            f"{meta.reference_amplitude:.6g}"
        )

    # ========================================================
    # ACTIVATION
    # ========================================================

    X_star = activation(
        u_star,
        axis=xi,
        resolution_scale=(
            meta.resolution_scale
        ),
        verbose=verbose,
    )

    # ========================================================
    # ACTIVITY
    # ========================================================

    A_star = activity(
        X_star,
        axis=xi,
        resolution_scale=(
            meta.resolution_scale
        ),
        verbose=verbose,
    )

    # ========================================================
    # TAU
    # ========================================================

    tau_eff = characteristic_time(
        tau=tau,
        metadata=meta.to_dict(),
        domain=meta.domain,
        system=meta.system_type,
        verbose=verbose,
    )

    meta.characteristic_time = tau_eff

    # ========================================================
    # REDUCED TIME
    # ========================================================

    eps = float(
        cfg.get(
            "epsilon",
            1e-12,
        )
    )

    t_star = (
        xi
        / max(tau_eff, eps)
    )

    if verbose:

        print(
            "[compute] "
            f"tau={tau_eff:.6g}"
        )

    # ========================================================
    # MEMORY
    # ========================================================

    M = memory(
        A_star,
        tau_eff,
        axis=xi,
        mechanism=meta.mechanism,
        domain=meta.domain,
        activity_factor=(
            meta.activity_factor
            if meta.activity_factor
            is not None
            else "auto"
        ),
        verbose=verbose,
    )

    # ========================================================
    # ORGANIZATION
    # ========================================================

    O = organization(
        X_star,
        tau_eff,
        axis=xi,
        mechanism=meta.mechanism,
        domain=meta.domain,
        activity_factor=(
            meta.activity_factor
            if meta.activity_factor
            is not None
            else "auto"
        ),
        verbose=verbose,
    )

    # ========================================================
    # INTENSITIES
    # ========================================================

    D, S = compute_intensities(
        X_star,
        A_star,
        M,
        O,
        verbose=verbose,
    )

    # ========================================================
    # BETA
    # ========================================================

    J, U, beta = compute_beta(
        D,
        S,
        M,
        O,
        verbose=verbose,
    )

    # ========================================================
    # POWER
    # ========================================================

    if P_c == "auto" or P_c is None:

        P_eff = characteristic_power(
            value=None,
            tau=tau_eff,
            system=meta.system_type,
            domain=meta.domain,
            A_ref=A_ref_used,
            verbose=verbose,
        )

    else:

        P_eff = _evaluate_power(
            P_c,
            xi,
        )

    meta.characteristic_power = (
        float(P_eff)
        if np.ndim(P_eff) == 0
        else None
    )

    # ========================================================
    # OBSERVABLE
    # ========================================================

    b = agencity(
        beta,
        P_eff,
        verbose=verbose,
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = AgencityResult(

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

        A_ref=float(
            meta.reference_amplitude
            if meta.reference_amplitude
            is not None
            else 1.0
        ),

        unit=meta.unit,

        observable_kind=meta.observable_kind,

        domain=meta.domain,

        metadata=meta,

        config=dict(cfg),
    )

    # ========================================================
    # DONE
    # ========================================================

    if verbose:
        print("[compute] Done")

    return result