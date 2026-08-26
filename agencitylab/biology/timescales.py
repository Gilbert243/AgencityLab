"""Biological reference-time and causal-memory calibration helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from agencitylab.models import ParameterProvenance, ParameterSource

from .models import (
    BiologicalMemoryReference,
    BiologicalReference,
    BiologicalTimeReference,
)

_REFERENCE_SPECIFIC_ENERGY_SI = 1.0  # J / kg, fixed by the biological mapping


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate sampled data with the trapezoid rule on supported NumPy floors."""
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _finite_1d(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def specific_metabolic_reference(
    standardized_powers: ArrayLike,
    masses: ArrayLike,
) -> float:
    """Return the reference median standardized power per unit mass.

    Inputs follow the biological SI convention: standardized sustainable
    metabolic powers are supplied in watts and masses in kilograms, giving
    ``q_reference`` in W/kg. AgencityLab performs no implicit unit conversion.
    """
    powers = _finite_1d(standardized_powers, name="standardized_powers")
    masses_array = _finite_1d(masses, name="masses")
    if powers.shape != masses_array.shape:
        raise ValueError("standardized_powers and masses must have the same shape")
    if np.any(powers <= 0.0):
        raise ValueError("standardized_powers must be strictly positive")
    if np.any(masses_array <= 0.0):
        raise ValueError("masses must be strictly positive")
    return float(np.median(powers / masses_array))


def biological_reference_time(
    standardized_powers: ArrayLike,
    masses: ArrayLike,
    *,
    reference: BiologicalReference,
    provenance: ParameterProvenance | None = None,
    source_ids: Sequence[str] = (),
) -> BiologicalTimeReference:
    """Construct the biological class/reference characteristic time ``tau``.

    The mapping defines ``tau = (1 J kg^-1) / q_reference`` with
    ``q_reference = median(P_std / mass)``. The 1 J/kg numerator is a fixed
    program normalization, not an inferred biological energy reserve.
    """
    powers = _finite_1d(standardized_powers, name="standardized_powers")
    masses_array = _finite_1d(masses, name="masses")
    if source_ids and len(source_ids) != powers.size:
        raise ValueError("source_ids must match the reference-cohort sample count")
    q_reference = specific_metabolic_reference(powers, masses_array)
    ratios = powers / masses_array
    tau = _REFERENCE_SPECIFIC_ENERGY_SI / q_reference
    dispersion = float(np.median(np.abs(ratios - np.median(ratios))))
    if provenance is None:
        provenance = ParameterProvenance(
            source=ParameterSource.DERIVED_MATHEMATICALLY,
            note="Biological reference time from median standardized metabolic power per mass",
            reference="Foundations of Biological Agencity (2026-08-26)",
        )
    return BiologicalTimeReference(
        reference=reference,
        q_reference=q_reference,
        tau=tau,
        sample_size=int(powers.size),
        dispersion=dispersion,
        unit="s",
        source_ids=tuple(source_ids),
        provenance=provenance,
    )


def causal_time(t: ArrayLike, h: ArrayLike) -> float:
    """Return the first absolute moment of an identified impulse response.

    ``theta = integral(t |h(t)| dt) / integral(|h(t)| dt)``.
    The time axis must be finite, strictly increasing and non-negative.
    """
    axis = _finite_1d(t, name="t")
    response = _finite_1d(h, name="h")
    if axis.shape != response.shape:
        raise ValueError("t and h must have the same shape")
    if axis.size < 2:
        raise ValueError("at least two impulse-response samples are required")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("t must be strictly increasing")
    if axis[0] < 0.0:
        raise ValueError("t must be non-negative")

    absolute_response = np.abs(response)
    area = _trapezoid(absolute_response, axis)
    first_moment = _trapezoid(axis * absolute_response, axis)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("impulse response must have a finite non-zero absolute area")
    if not np.isfinite(first_moment) or first_moment < 0.0:
        raise ValueError("impulse response must have a finite first absolute moment")
    theta = first_moment / area
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("causal time must be strictly positive and finite")
    return float(theta)


def calibrate_memory_reference(
    impulse_responses: Iterable[tuple[ArrayLike, ArrayLike]],
    *,
    reference: BiologicalReference,
    provenance: ParameterProvenance | None = None,
    time_unit: str = "s",
    source_ids: Sequence[str] = (),
) -> BiologicalMemoryReference:
    """Calibrate the common biological CRM depth from independent responses.

    Each item is ``(t, h)`` for one identified calibration impulse response.
    The biological mapping uses the arithmetic population mean of individual
    causal times as the common reference ``w(R,s)``. Median/MAD information is
    retained only as a robustness descriptor.
    """
    time_unit = str(time_unit).strip()
    if not time_unit:
        raise ValueError("time_unit must be non-empty")
    response_items = tuple(impulse_responses)
    if source_ids and len(source_ids) != len(response_items):
        raise ValueError("source_ids must match the impulse-response count")
    times = tuple(causal_time(t, h) for t, h in response_items)
    if not times:
        raise ValueError("at least one impulse response is required")
    values = np.asarray(times, dtype=float)
    w = float(np.mean(values))
    dispersion = float(np.median(np.abs(values - np.median(values))))
    if provenance is None:
        provenance = ParameterProvenance(
            source=ParameterSource.DERIVED_MATHEMATICALLY,
            note="Biological CRM depth from the mean causal time of calibration responses",
            reference="Foundations of Biological Agencity (2026-08-26)",
        )
    return BiologicalMemoryReference(
        reference=reference,
        w=w,
        individual_causal_times=times,
        dispersion=dispersion,
        unit=time_unit,
        source_ids=tuple(source_ids),
        provenance=provenance,
    )
