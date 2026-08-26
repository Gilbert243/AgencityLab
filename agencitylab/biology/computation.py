"""Thin biological orchestration over the public AgencityLab computation API."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from agencitylab.exceptions import AgencityValidationError, PhysicalParameterError

from .models import (
    BiologicalAgencityResult,
    BiologicalObservable,
    FrozenBiologicalProtocol,
    ObservableBlock,
)
from .validation import (
    validate_observable_against_protocol,
    validate_protocol,
    validate_reference_amplitudes,
)


def _validate_power_partition(
    P_c_components: ArrayLike,
    *,
    n_components: int,
    P_c_total: float,
) -> np.ndarray:
    powers = np.asarray(P_c_components, dtype=float)
    if powers.ndim != 1 or powers.size != n_components:
        raise PhysicalParameterError(
            "P_c_components must contain exactly one explicit capacity per component"
        )
    if not np.all(np.isfinite(powers)) or np.any(powers < 0.0):
        raise PhysicalParameterError("P_c_components must contain non-negative finite values")
    if not np.isclose(float(np.sum(powers)), P_c_total, rtol=1e-9, atol=1e-12):
        raise PhysicalParameterError(
            "P_c_components must form a physical partition whose sum equals organism P_c"
        )
    return powers


def compute_biological_agencity(
    observable: BiologicalObservable,
    protocol: FrozenBiologicalProtocol,
    *,
    P_c_components: ArrayLike | None = None,
) -> BiologicalAgencityResult:
    """Compute Agencity from a frozen biological mapping.

    A scalar biological observable delegates to the canonical public
    :func:`agencitylab.api.compute_agencity`. A multi-coordinate observable
    delegates to :func:`agencitylab.api.compute_multivariate_agencity` only when
    an explicit physical ``P_c`` partition is supplied. The organism capacity is
    never silently broadcast once per measurement coordinate.
    """
    from agencitylab.api import compute_agencity, compute_multivariate_agencity

    biology_protocol = protocol.protocol
    validation = validate_protocol(biology_protocol)
    if not validation.valid:
        raise AgencityValidationError(
            "frozen biology protocol is invalid: " + "; ".join(validation.errors)
        )
    validate_observable_against_protocol(observable, biology_protocol)
    raw_A_ref = biology_protocol.reference_amplitudes
    if any(value is None for value in raw_A_ref):
        raise PhysicalParameterError("biology protocol contains unresolved A_ref values")
    resolved_A_ref = tuple(float(value) for value in raw_A_ref if value is not None)
    A_ref = validate_reference_amplitudes(
        resolved_A_ref,
        n_components=observable.n_components,
    )
    tau = biology_protocol.time_reference.tau
    w = biology_protocol.memory_reference.w
    P_c = biology_protocol.capacity.value

    if observable.n_components == 1:
        if P_c_components is not None:
            powers = _validate_power_partition(
                P_c_components,
                n_components=1,
                P_c_total=P_c,
            )
            if not np.isclose(powers[0], P_c):
                raise PhysicalParameterError("single-coordinate P_c partition must equal P_c")
        coordinate = biology_protocol.coordinates[0]
        observable_block = (
            coordinate.block
            if isinstance(coordinate.block, ObservableBlock)
            else ObservableBlock(coordinate.block)
        )
        result = compute_agencity(
            observable.values[:, 0],
            observable.xi,
            A_ref=float(A_ref[0]),
            tau=tau,
            w=w,
            P_c=P_c,
            unit=coordinate.unit,
            coordinate_unit=observable.coordinate_unit,
            power_unit=biology_protocol.capacity.unit,
            observable_kind=observable_block.value,
            domain="biology",
            system_type=biology_protocol.reference.support,
            metadata={
                "extra": {
                    "biology_protocol_id": biology_protocol.protocol_id,
                    "biology_protocol_version": biology_protocol.version,
                    "biology_protocol_hash": protocol.content_hash,
                    "biology_reference": biology_protocol.reference.address,
                    "biology_subject_id": biology_protocol.capacity.subject_id,
                    "biology_scientific_status": "experimental",
                }
            },
        )
        return BiologicalAgencityResult(
            subject_id=biology_protocol.capacity.subject_id,
            protocol_hash=protocol.content_hash,
            mode="scalar",
            result=result,
        )

    if P_c_components is None:
        raise PhysicalParameterError(
            "multi-coordinate biological Agencity requires an explicit physical P_c_components "
            "partition; organism P_c is not broadcast to observable coordinates"
        )
    powers = _validate_power_partition(
        P_c_components,
        n_components=observable.n_components,
        P_c_total=P_c,
    )
    result = compute_multivariate_agencity(
        observable.values,
        observable.xi,
        A_ref=A_ref,
        tau=tau,
        w=w,
        P_c=powers,
        sample_axis=0,
    )
    return BiologicalAgencityResult(
        subject_id=biology_protocol.capacity.subject_id,
        protocol_hash=protocol.content_hash,
        mode="multivariate",
        result=result,
        P_c_components=tuple(float(item) for item in powers),
    )
