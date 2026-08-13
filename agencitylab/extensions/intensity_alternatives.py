"""Historical intensity candidates from Volume 2 Chapter 14.

These functions expose source-defined alternatives that were examined before the
canonical logarithmic contrast was retained. They are experimental reference
formulas only: none of them replaces ``J = log((e + D) / (e + S))`` in the
canonical ``u -> beta -> b`` pipeline.

The source contains an unresolved collision between the offset form printed at
the end of Section 14.3 and the ``I3`` expression printed in Section 14.4: the
two displayed formulas are algebraically identical. AgencityLab therefore
exposes that printed expression once, under a neutral name, instead of inventing
a distinction that is not present in the accepted document.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus


INTENSITY_ALTERNATIVES_STATUS = ScientificStatus.EXPERIMENTAL


def _finite_broadcast(*values, names: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if len(arrays) != len(names):  # pragma: no cover - internal contract
        raise RuntimeError("names must match values")
    for array, name in zip(arrays, names, strict=True):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    return tuple(np.broadcast_arrays(*arrays))


def sum_intensity(X, A, M, O):
    """Return the rejected Chapter-14 sum intensity ``I1``.

    ``I1 = |X| + |A X| + |M| + |O|``.
    """
    X_arr, A_arr, M_arr, O_arr = _finite_broadcast(
        X,
        A,
        M,
        O,
        names=("X", "A", "M", "O"),
    )
    return np.abs(X_arr) + np.abs(A_arr * X_arr) + np.abs(M_arr) + np.abs(O_arr)


def sum_log_intensity(X, A, M, O):
    """Return ``J1 = log(I1)`` and preserve the source singularity at ``I1=0``.

    At exact rest ``I1 = 0`` and the historical formula is singular. The
    function returns ``-inf`` there rather than inserting an epsilon.
    """
    intensity = np.asarray(sum_intensity(X, A, M, O), dtype=float)
    if intensity.ndim == 0:
        value = float(intensity)
        return float(np.log(value)) if value > 0.0 else -np.inf

    out = np.full(intensity.shape, -np.inf, dtype=float)
    positive = intensity > 0.0
    out[positive] = np.log(intensity[positive])
    return out


def raw_ratio_intensity(X, A, M, O):
    """Return the unregularised ratio candidate of Equation (14.2).

    ``I2 = (|X| + |A X|) / (|M| + |O|)``.

    The source denominator singularity is exposed explicitly: a positive
    numerator over zero returns ``+inf`` and ``0/0`` returns ``NaN``. No
    numerical epsilon is introduced.
    """
    X_arr, A_arr, M_arr, O_arr = _finite_broadcast(
        X,
        A,
        M,
        O,
        names=("X", "A", "M", "O"),
    )
    numerator = np.abs(X_arr) + np.abs(A_arr * X_arr)
    denominator = np.abs(M_arr) + np.abs(O_arr)

    if numerator.ndim == 0:
        num = float(numerator)
        den = float(denominator)
        if den > 0.0:
            return num / den
        return np.inf if num > 0.0 else np.nan

    out = np.full(numerator.shape, np.nan, dtype=float)
    defined = denominator > 0.0
    out[defined] = numerator[defined] / denominator[defined]
    singular_positive = (denominator == 0.0) & (numerator > 0.0)
    out[singular_positive] = np.inf
    return out


def printed_offset_ratio_candidate(X, A, M, O):
    """Return the offset expression printed in both Sections 14.3 and 14.4.

    The accepted source prints

    ``e + (|X| + |A X|) / (e + |M| + |O|)``

    first as the regularised ratio candidate and then again as ``I3``. This
    helper implements the common printed expression exactly with ``e = exp(1)``
    while deliberately making no claim that the two source labels are distinct.
    """
    X_arr, A_arr, M_arr, O_arr = _finite_broadcast(
        X,
        A,
        M,
        O,
        names=("X", "A", "M", "O"),
    )
    numerator = np.abs(X_arr) + np.abs(A_arr * X_arr)
    denominator = np.e + np.abs(M_arr) + np.abs(O_arr)
    out = np.e + numerator / denominator
    return out.item() if out.ndim == 0 else out


__all__ = [
    "INTENSITY_ALTERNATIVES_STATUS",
    "printed_offset_ratio_candidate",
    "raw_ratio_intensity",
    "sum_intensity",
    "sum_log_intensity",
]
