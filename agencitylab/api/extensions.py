"""Public multiscale, discrete, multivariate, and geometric extension APIs."""

from __future__ import annotations

from typing import Any

import numpy as np

from agencitylab.core.agencity import agencity
from agencitylab.core.beta import compute_beta
from agencitylab.core.discrete import volume2_first_difference, volume2_second_difference
from agencitylab.core.intensity import compute_intensities
from agencitylab.core.memory import memory
from agencitylab.core.multiscale import (
    agencity_spectrum,
    multivariate_agencity,
    optimize_memory_window,
)
from agencitylab.core.organization import organization
from agencitylab.core.validation import validate_positive_scalar, validate_signal
from agencitylab.models import AgencityResult, ExperimentMetadata

from .compute import compute_agencity

RIEMANNIAN_EXTENSION_STATUS = (
    "experimental: intrinsic kinematic primitives implemented; detailed analysis is deferred"
)


def compute_agencity_spectrum(
    u,
    xi,
    taus,
    *,
    A_ref,
    P_c,
    windows=None,
    return_full: bool = False,
):
    """Compute the time-resolved ``b(t, tau)`` spectrum.

    With ``windows=None`` each scale uses the implementation fallback convention
    ``w=tau``. Supplying a scalar or sequence keeps ``w`` explicit and
    independent, as defined by Volume 2 of the Theory of Agencity.
    """
    return agencity_spectrum(
        xi,
        u,
        taus,
        A_ref=A_ref,
        P_c=P_c,
        windows=windows,
        return_full=return_full,
    )


def optimize_agencity_window(
    u,
    xi,
    *,
    tau,
    A_ref,
    P_c,
    candidates=None,
    n_candidates: int = 24,
):
    """Select the CRM width using the theory's Chapter 13 ``Phi2`` criterion."""
    return optimize_memory_window(
        xi,
        u,
        tau=tau,
        A_ref=A_ref,
        P_c=P_c,
        candidates=candidates,
        n_candidates=n_candidates,
    )


def compute_multivariate_agencity(
    u,
    xi,
    *,
    A_ref,
    tau,
    P_c,
    w=None,
    sample_axis: int = 0,
):
    """Compute the theory's Pc-weighted multivariate construction componentwise."""
    return multivariate_agencity(
        xi,
        u,
        A_ref=A_ref,
        tau=tau,
        P_c=P_c,
        w=w,
        sample_axis=sample_axis,
    )


def compute_discrete_agencity(
    u,
    *,
    delta,
    A_ref,
    tau,
    P_c,
    w=None,
    t0: float = 0.0,
    unit: str | None = None,
    coordinate_unit: str | None = None,
    power_unit: str | None = None,
    observable_kind: str | None = None,
    domain: str | None = None,
    mechanism: str | None = None,
    system_type: str | None = None,
    environment: str | None = None,
    geometry: str | None = None,
    metadata: dict[str, Any] | ExperimentMetadata | None = None,
    verbose: bool = False,
) -> AgencityResult:
    """Compute the explicit uniformly sampled discrete construction of Volume 2.

    Interior activation and activity use the Volume-2 stencils directly rather
    than the successive ``gradient -> gradient`` approximation used when the
    continuous reference pipeline is sampled numerically:

    ``X_n = (u[n+1] - u[n-1]) / (2 delta)``

    ``A_n = (u[n+1] - 2 u[n] + u[n-1]) / delta**2``.

    AgencityLab applies these stencils to the canonical reduced observable
    ``u_star`` on reduced spacing ``delta_star = delta / tau``. Boundaries use
    documented second-order one-sided formulas so result length is preserved.
    The CRM, intensities, ``J``, ``U``, ``beta`` and ``b`` then use the same core
    operators as the scalar theory.

    The ordinary :func:`compute_agencity` function remains the sole reference
    canonical continuous ``u -> b`` pipeline. This function is the explicit
    Volume-2 discrete extension and is not an alias for ``gradient(gradient(u))``.
    """
    values = validate_signal(u, name="u")
    if values.ndim != 1:
        raise ValueError("u must be one-dimensional for the discrete scalar API")
    delta = validate_positive_scalar(delta, name="delta")
    t0 = float(t0)
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite")
    xi = t0 + delta * np.arange(values.size, dtype=float)

    # Reuse the stable public path only for physical/context resolution and the
    # canonical non-derivative data model. The derivative-dependent arrays are
    # recomputed below from the explicit Volume-2 stencils.
    base = compute_agencity(
        u=values,
        xi=xi,
        A_ref=A_ref,
        tau=tau,
        w=w,
        P_c=P_c,
        unit=unit,
        coordinate_unit=coordinate_unit,
        power_unit=power_unit,
        observable_kind=observable_kind,
        domain=domain,
        mechanism=mechanism,
        system_type=system_type,
        environment=environment,
        geometry=geometry,
        metadata=metadata,
        verbose=verbose,
    )

    delta_star = delta / base.tau
    X_star = volume2_first_difference(base.u_star, delta_star)
    A_star = volume2_second_difference(base.u_star, delta_star)

    if np.all(values == values[0]):
        M = base.M
        O = base.O
        D = base.D
        S = base.S
        J = base.J
        U = base.U
        beta = base.beta
        b = base.b
    else:
        memory_window = base.memory_window
        M = memory(base.u_star, base.tau, axis=base.xi, window=memory_window)
        O = organization(
            base.u_star,
            X_star,
            base.tau,
            axis=base.xi,
            window=memory_window,
        )
        D, S = compute_intensities(X_star, A_star, M, O)
        J, U, beta = compute_beta(D, S, M, O)
        b = agencity(beta, base.P_c)

    metadata_model = ExperimentMetadata.from_dict(base.metadata)
    metadata_model.extra.update(
        {
            "discrete_formulation": "Volume 2 explicit centered differences",
            "discrete_derivative_scheme": "explicit centered Volume-2 stencils",
            "discrete_boundary_scheme": (
                "second-order one-sided boundaries; three-point fallback for N=3"
            ),
            "discrete_delta": float(delta),
            "discrete_delta_star": float(delta_star),
        }
    )

    return AgencityResult(
        xi=base.xi,
        u=base.u,
        u_star=base.u_star,
        X_star=X_star,
        A_star=A_star,
        t_star=base.t_star,
        tau=base.tau,
        P_c=base.P_c,
        A_ref=base.A_ref,
        M=M,
        O=O,
        D=D,
        S=S,
        J=J,
        U=U,
        beta=beta,
        b=b,
        theta=np.angle(U),
        unit=base.unit,
        coordinate_unit=base.coordinate_unit,
        power_unit=base.power_unit,
        observable_kind=base.observable_kind,
        domain=base.domain,
        system_type=base.system_type,
        mechanism=base.mechanism,
        metadata=metadata_model,
    )


def riemannian_extension_status() -> dict:
    """Report the implementation boundary of the Riemannian construction.

    Volume 2 defines covariant velocity/acceleration and the intrinsic dynamic
    intensity, while explicitly deferring the detailed analysis. AgencityLab
    implements those source-defined kinematic primitives under
    ``agencitylab.extensions`` but does not invent the missing general
    CRM/vector-state manifold pipeline.
    """
    return {
        "status": RIEMANNIAN_EXTENSION_STATUS,
        "implemented": False,
        "intrinsic_primitives_implemented": True,
        "full_pipeline_implemented": False,
        "reason": (
            "Definition 12.4 kinematic primitives are implemented; the source explicitly "
            "defers detailed Riemannian Agencity analysis, so no full pipeline is fabricated"
        ),
    }
