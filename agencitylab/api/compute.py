"""Public orchestration for the canonical Agencity ``u -> b`` pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from agencitylab.core.activation import activation, reduced_coordinate
from agencitylab.core.activity import activity
from agencitylab.core.agencity import agencity
from agencitylab.core.beta import compute_beta
from agencitylab.core.intensity import compute_intensities
from agencitylab.core.memory import memory
from agencitylab.core.organization import organization
from agencitylab.core.validation import is_exactly_constant
from agencitylab.exceptions import AgencityValidationError
from agencitylab.models import AgencityResult

from ._compute_support import (
    MetadataInput,
    PowerInput,
    prepare_metadata,
    resolve_characteristic_power,
    resolve_characteristic_time,
    resolve_memory_window,
    resolve_normalized_observable,
)
from .validation import prepare_inputs


def compute_agencity(
    u: ArrayLike,
    xi: ArrayLike | None = None,
    *,
    A_ref: float | str | None = None,
    tau: float | str | None = "auto",
    w: float | None = None,
    P_c: PowerInput = "auto",
    unit: str | None = None,
    coordinate_unit: str | None = None,
    power_unit: str | None = None,
    observable_kind: str | None = None,
    domain: str | None = None,
    mechanism: str | None = None,
    system_type: str | None = None,
    environment: str | None = None,
    geometry: str | None = None,
    metadata: MetadataInput = None,
    verbose: bool = False,
) -> AgencityResult:
    """Compute the reference canonical scalar-signal Theory of Agencity pipeline.

    Parameters
    ----------
    u:
        Finite one-dimensional observable samples.
    xi:
        Strictly increasing coordinates. If omitted, sample indices are used.
    A_ref, tau, w, P_c:
        Physical/contextual parameters. ``A_ref``, ``tau`` and scalar ``P_c`` may
        use an explicitly registered contextual convention when set to ``"auto"``.
        ``P_c`` may also be a non-negative sampled profile or a callable evaluated
        on ``xi``. No signal statistic is used to invent these physical values.
    unit, coordinate_unit, power_unit:
        Descriptive labels only; AgencityLab performs no implicit unit conversion.
    metadata:
        Optional reproducibility metadata.
    verbose:
        Emit diagnostic progress from the canonical numerical stages.

    Notes
    -----
    The CRM width ``w`` is distinct from ``tau``. If ``w`` is omitted, the
    implementation convention ``w = tau`` is recorded explicitly in metadata.

    The 1.0 canonical reference pipeline is NumPy based. Experimental JAX and
    Numba primitives live under :mod:`agencitylab.backends`; selecting one does
    not masquerade as a different canonical computation backend.
    """
    xi_was_provided = xi is not None
    xi_array, u_array = prepare_inputs(u=u, xi=xi)

    metadata_model = prepare_metadata(
        metadata,
        xi_was_provided=xi_was_provided,
        unit=unit,
        coordinate_unit=coordinate_unit,
        power_unit=power_unit,
        observable_kind=observable_kind,
        domain=domain,
        mechanism=mechanism,
        system_type=system_type,
        environment=environment,
        geometry=geometry,
        A_ref=A_ref,
    )

    u_star, A_ref_used = resolve_normalized_observable(
        u_array,
        A_ref=A_ref,
        metadata=metadata_model,
        verbose=verbose,
    )
    tau_eff = resolve_characteristic_time(
        tau,
        metadata=metadata_model,
        verbose=verbose,
    )
    memory_window = resolve_memory_window(
        w,
        tau=tau_eff,
        metadata=metadata_model,
    )
    P_eff = resolve_characteristic_power(
        P_c,
        xi=xi_array,
        tau=tau_eff,
        metadata=metadata_model,
        verbose=verbose,
    )

    t_star = reduced_coordinate(xi_array, tau_eff)

    if is_exactly_constant(u_array):
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
                axis=xi_array,
                window=memory_window,
                verbose=verbose,
            )
            O = organization(
                u_star,
                X_star,
                tau_eff,
                axis=xi_array,
                window=memory_window,
                verbose=verbose,
            )
            D, S = compute_intensities(X_star, A_star, M, O, verbose=verbose)
            J, U, beta = compute_beta(D, S, M, O, verbose=verbose)
            b = agencity(beta, P_eff, verbose=verbose)
        except ValueError as exc:
            raise AgencityValidationError(f"numerical pipeline failed: {exc}") from exc

    return AgencityResult(
        xi=xi_array,
        u=u_array,
        u_star=u_star,
        X_star=X_star,
        A_star=A_star,
        t_star=t_star,
        tau=tau_eff,
        P_c=P_eff,
        A_ref=A_ref_used,
        M=M,
        O=O,
        D=D,
        S=S,
        J=J,
        U=U,
        beta=beta,
        b=b,
        unit=metadata_model.unit,
        coordinate_unit=metadata_model.coordinate_unit,
        power_unit=metadata_model.power_unit,
        observable_kind=metadata_model.observable_kind,
        domain=metadata_model.domain,
        system_type=metadata_model.system_type,
        mechanism=metadata_model.mechanism,
        metadata=metadata_model,
    )
