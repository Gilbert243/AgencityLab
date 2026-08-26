"""Scientific-contract validation for biological Agencity protocols."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .models import BiologicalObservable, BiologicalProtocol, ScientificValidation


def validate_reference_amplitudes(A_ref: ArrayLike, *, n_components: int) -> np.ndarray:
    """Validate explicit component reference amplitudes without estimating them."""
    values = np.asarray(A_ref, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.size != n_components:
        raise ValueError("A_ref must contain exactly one value per observable coordinate")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all A_ref values must be strictly positive and finite")
    return values


def validate_protocol(protocol: BiologicalProtocol) -> ScientificValidation:
    """Validate the biological mapping contract without judging the theory."""
    errors: list[str] = []
    warnings: list[str] = []

    unresolved = [
        coordinate.name for coordinate in protocol.coordinates if coordinate.A_ref is None
    ]
    if unresolved:
        errors.append("unresolved_A_ref:" + ",".join(unresolved))

    test_ids = protocol.dataset_split.test
    tau_leakage = set(protocol.time_reference.source_ids) & test_ids
    if tau_leakage:
        errors.append("tau_reference_uses_test_samples:" + ",".join(sorted(tau_leakage)))
    w_leakage = set(protocol.memory_reference.source_ids) & test_ids
    if w_leakage:
        errors.append("w_reference_uses_test_samples:" + ",".join(sorted(w_leakage)))

    overlaps = protocol.dataset_split.overlaps()
    if overlaps:
        details = ";".join(
            f"{name}={','.join(sorted(values))}" for name, values in overlaps.items()
        )
        errors.append("dataset_independence_violation:" + details)

    if protocol.target_informed_calibration:
        errors.append("target_informed_calibration_is_not_admissible")
    if protocol.time_reference.unit != protocol.memory_reference.unit:
        errors.append("tau_and_w_time_units_do_not_match")

    if not protocol.capacity.challenge_protocol:
        warnings.append("capacity_challenge_protocol_not_documented")
    if not protocol.capacity.measurement_method:
        warnings.append("capacity_measurement_method_not_documented")
    if not protocol.measurement_map_id:
        warnings.append("measurement_map_not_attached_to_protocol")
    if np.isclose(protocol.time_reference.tau, protocol.memory_reference.w):
        warnings.append("tau_and_w_are_numerically_equal_but_remain_distinct_references")

    return ScientificValidation(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))


def validate_observable_against_protocol(
    observable: BiologicalObservable,
    protocol: BiologicalProtocol,
) -> None:
    """Require coordinate identity, units and measurement-map provenance to match."""
    if observable.n_components != len(protocol.coordinates):
        raise ValueError("observable component count does not match biology protocol")
    for actual, expected in zip(observable.coordinates, protocol.coordinates):
        if (
            actual.name != expected.name
            or actual.block != expected.block
            or actual.unit != expected.unit
            or actual.A_ref != expected.A_ref
        ):
            raise ValueError(
                "observable coordinate definitions must match the frozen biology protocol"
            )
    if observable.coordinate_unit != protocol.time_reference.unit:
        raise ValueError("observable time unit does not match biological tau/w reference unit")

    if observable.xi.size < 3:
        raise ValueError("biological observable time axis must contain at least three samples")
    diffs = np.diff(observable.xi)
    if np.any(diffs <= 0.0):
        raise ValueError("biological observable time axis must be strictly increasing")
    step = float(diffs[0])
    tolerance = float(np.finfo(float).eps * max(1.0, abs(step)) * 64.0)
    if not np.allclose(diffs, step, rtol=1e-10, atol=tolerance):
        raise ValueError("biological CRM requires a uniformly sampled observable time axis")

    w = protocol.memory_reference.w
    window_samples = int(round(w / step))
    if window_samples < 1:
        raise ValueError(
            "biological w is smaller than one sampling interval; increase temporal resolution"
        )
    represented_w = window_samples * step
    representation_tolerance = max(
        float(np.finfo(float).eps) * max(1.0, abs(w)) * 128.0,
        abs(step) * 1e-9,
    )
    if not np.isclose(represented_w, w, rtol=1e-9, atol=representation_tolerance):
        raise ValueError(
            "biological w must be an integer multiple of the observable sampling interval"
        )
    if observable.xi.size < 2 * window_samples:
        raise ValueError("observable is too short for two biological CRM windows")
    if protocol.measurement_map_id:
        if observable.measurement_map_id != protocol.measurement_map_id:
            raise ValueError("observable measurement map id does not match biology protocol")
        if observable.measurement_map_version != protocol.measurement_map_version:
            raise ValueError("observable measurement map version does not match biology protocol")
