"""Topology diagnostics for the autonomous complex field ``phi``.

These diagnostics concern spatial field topology and are intentionally distinct
from any temporal winding diagnostic for the canonical ``beta(t)`` curve in
``agencitylab.analysis``. They do not implement or redefine a real-agencity
criterion.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_complex_contour(values) -> np.ndarray:
    contour = np.asarray(values)
    if contour.ndim != 1:
        raise ValueError("phi_contour must be one-dimensional and ordered around a closed contour")
    if contour.size < 3:
        raise ValueError("phi_contour must contain at least three samples")
    if not np.issubdtype(contour.dtype, np.number) or np.issubdtype(contour.dtype, np.bool_):
        raise TypeError("phi_contour must contain numeric values")
    contour = np.asarray(contour, dtype=complex)
    if not np.all(np.isfinite(contour)):
        raise ValueError("phi_contour must contain only finite values")
    if np.any(np.abs(contour) == 0.0):
        raise ValueError("phase winding is undefined when the contour intersects an exact field zero")
    return contour


def phase_winding(phi_contour) -> float:
    """Return the numerical phase winding around an ordered closed contour.

    The diagnostic evaluates the total unwrapped phase change divided by
    ``2*pi``. The first sample does not need to be repeated at the end: the
    contour is closed numerically by appending its first phase before applying
    ``numpy.unwrap``.

    The result is returned as a float rather than being silently rounded to an
    integer. For a sufficiently resolved contour that avoids field zeros, a
    topological vortex gives a value close to its integer winding number. No
    amplitude threshold and no real-agencity criterion are used here.
    """

    contour = _finite_complex_contour(phi_contour)
    phases = np.angle(contour)
    closed_phases = np.concatenate((phases, phases[:1]))
    unwrapped = np.unwrap(closed_phases)
    return float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))


def field_zero_mask(phi, *, tolerance: float | None = None) -> np.ndarray:
    """Locate exact or caller-defined near-zero regions of a complex field.

    With ``tolerance=None`` only exact numerical zeros ``|phi| == 0`` are
    detected. A near-zero tolerance is never invented by the library: callers
    must provide it explicitly, and it must be finite and non-negative.
    """

    field = np.asarray(phi)
    if field.size == 0:
        raise ValueError("phi must not be empty")
    if not np.issubdtype(field.dtype, np.number) or np.issubdtype(field.dtype, np.bool_):
        raise TypeError("phi must contain numeric values")
    if not np.all(np.isfinite(field)):
        raise ValueError("phi must contain only finite values")

    magnitude = np.abs(field)
    if tolerance is None:
        return magnitude == 0.0

    try:
        threshold = float(tolerance)
    except Exception as exc:
        raise ValueError("tolerance must be a finite non-negative real scalar") from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return magnitude <= threshold
