"""Objective CRM-window criteria from Volume 2 Chapter 13.

The existing automatic selector follows Appendix C and minimises ``Phi2``
(angular stability).  This module adds the other two source-defined criteria
without changing that default algorithm:

- ``Phi1``: mean absolute logarithmic contrast, to maximise;
- ``Phi3``: entropy of discretised structural orientations, to maximise.

For ``Phi3`` the theory defines frequencies of discretised angle bins but does
not prescribe a universal bin count or bin edges.  The caller must therefore
supply explicit edges; AgencityLab does not invent a universal discretisation.

Scientific status: experimental extension/selection criterion.  These helpers
must not be used to silently infer a physically specified ``tau`` or ``w``.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.EXPERIMENTAL


def _finite_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def mean_contrast_criterion(contrast, *, coordinates=None) -> float:
    """Evaluate the Chapter-13 ``Phi1`` mean absolute contrast criterion.

    If ``coordinates`` is omitted, the arithmetic mean is the uniform-sampling
    numerical approximation to ``(1/T) integral |J_w(t)| dt``.  If coordinates
    are supplied they must be strictly increasing and the trapezoidal integral
    is divided by the represented duration.
    """

    values = _finite_1d(contrast, name="contrast")
    magnitude = np.abs(values)
    if coordinates is None:
        return float(np.mean(magnitude))

    coordinate = _finite_1d(coordinates, name="coordinates")
    if coordinate.shape != values.shape:
        raise ValueError("coordinates must have the same shape as contrast")
    increments = np.diff(coordinate)
    if np.any(increments <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    duration = coordinate[-1] - coordinate[0]
    return float(np.trapezoid(magnitude, coordinate) / duration)


def orientational_entropy_criterion(
    theta,
    *,
    bin_edges,
    valid_mask=None,
) -> float:
    """Evaluate the Chapter-13 ``Phi3 = -sum p_k ln(p_k)`` criterion.

    ``bin_edges`` are mandatory because Volume 2 does not prescribe a
    universal angular discretisation.  ``valid_mask`` may explicitly exclude
    samples where structural orientation is physically undefined (for example
    where ``S = 0``).  No epsilon is inserted into zero-frequency bins; they
    simply contribute zero to the Shannon sum by continuity.
    """

    angles = _finite_1d(theta, name="theta")
    edges = _finite_1d(bin_edges, name="bin_edges")
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must contain at least two strictly increasing edges")

    if valid_mask is None:
        selected = angles
    else:
        mask = np.asarray(valid_mask)
        if mask.shape != angles.shape or mask.dtype != np.bool_:
            raise ValueError("valid_mask must be a boolean array with the same shape as theta")
        selected = angles[mask]
    if selected.size == 0:
        raise ValueError("at least one structurally valid orientation is required")
    if np.any(selected < edges[0]) or np.any(selected > edges[-1]):
        raise ValueError("all selected orientations must lie within the supplied bin edges")

    counts, _ = np.histogram(selected, bins=edges)
    probabilities = counts[counts > 0].astype(float) / float(selected.size)
    return float(-np.sum(probabilities * np.log(probabilities)))
