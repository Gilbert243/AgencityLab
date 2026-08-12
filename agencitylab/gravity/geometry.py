"""Small numerical geometry primitives for classical Agencity gravity.

The Gravity layer follows Volume 2, Chapter 19 and therefore uses the
Lorentzian metric signature ``(-, +, +, +)``.  This is intentionally distinct
from the ``(+, -, -, -)`` flat-field convention used in Chapter 16 and by
``agencitylab.fields.dynamics``.  No conversion between those conventions is
performed implicitly here.

Scientific status: research.  These helpers are numerical tensor evaluators,
not a general-relativity solver or symbolic differential-geometry framework.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH
GRAVITY_METRIC_SIGNATURE = (-1, 1, 1, 1)


def _finite_real_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.any(np.imag(array) != 0.0):
            raise ValueError(f"{name} must be real")
        array = np.real(array)
    try:
        result = np.asarray(array, dtype=float)
    except Exception as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_numeric_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain real or complex numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _metric_array(metric, *, name: str = "metric") -> np.ndarray:
    array = _finite_real_array(metric, name=name)
    if array.ndim < 2 or array.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape (..., 4, 4)")
    return array


def _metric_for_shape(metric, base_shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = _metric_array(metric, name=name)
    expected = base_shape + (4, 4)
    if array.shape == (4, 4):
        return np.broadcast_to(array, expected)
    if array.shape != expected:
        raise ValueError(
            f"{name} shape {array.shape} is incompatible with field shape {base_shape}; "
            f"expected (4, 4) or {expected}"
        )
    return array


def _real_scalar_or_field(value, base_shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = _finite_real_array(value, name=name)
    if array.shape == ():
        return np.broadcast_to(array, base_shape)
    if array.shape != base_shape:
        raise ValueError(
            f"{name} shape {array.shape} is incompatible with field shape {base_shape}; "
            f"expected a scalar or {base_shape}"
        )
    return array


def _real_if_numerically_close(value, *, name: str):
    """Return a theoretically real quantity after checking roundoff only.

    This check does not alter a physical equation.  It rejects a significant
    imaginary component and removes only a component consistent with machine
    roundoff after the real-valued mathematical contraction has been evaluated.
    """

    array = np.asarray(value)
    if not np.iscomplexobj(array):
        return array
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(np.abs(np.imag(array)) > tolerance):
        raise ValueError(f"{name} is expected to be real but has a significant imaginary part")
    return np.real(array)


def minkowski_metric() -> np.ndarray:
    """Return the Chapter-19 Minkowski metric with signature ``(-,+,+,+)``."""

    return np.diag(np.array([-1.0, 1.0, 1.0, 1.0]))


def minkowski_inverse_metric() -> np.ndarray:
    """Return the inverse Chapter-19 Minkowski metric.

    For the diagonal ``(-,+,+,+)`` convention the matrix is its own inverse.
    """

    return minkowski_metric()


def sqrt_minus_g(metric):
    """Return ``sqrt(-det(g_mu_nu))`` for a finite Lorentzian metric array.

    The metric must have shape ``(..., 4, 4)`` and a strictly negative
    determinant at every supplied point.  No metric is inferred from field data.
    """

    array = _metric_array(metric)
    determinant = np.linalg.det(array)
    if not np.all(np.isfinite(determinant)):
        raise ValueError("metric determinant must be finite")
    if np.any(determinant >= 0.0):
        raise ValueError("metric must have negative determinant for sqrt(-g)")
    result = np.sqrt(-determinant)
    return float(result) if np.ndim(result) == 0 else result


def metric_with_perturbation(background_metric, perturbation) -> np.ndarray:
    """Return ``g_mu_nu = eta_mu_nu + h_mu_nu`` with explicit shape checks.

    This is only an algebraic representation of the linearisation discussed in
    Chapter 19.  It does not implement gravitational-wave evolution.
    """

    h = _metric_array(perturbation, name="perturbation")
    base_shape = h.shape[:-2]
    background = _metric_for_shape(background_metric, base_shape, name="background_metric")
    result = background + h
    if not np.all(np.isfinite(result)):
        raise ValueError("perturbed metric must remain finite")
    return result
