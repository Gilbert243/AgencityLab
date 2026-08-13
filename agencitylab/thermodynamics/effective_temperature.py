"""Temperature-dependent thermodynamic coefficients.

Scientific status: research. Physical temperatures are explicit contextual
inputs; this module does not estimate temperature from a signal or field.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_real_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def temperature_dependent_lambda(temperature, a, t_c):
    """Return ``lambda(T) = a * (T_c - T)`` from Volume 2, Section 18.4.

    No sign constraint is imposed on ``a`` because the source formula itself
    does not define such a software-domain restriction. For ``a > 0``, the
    mathematical consequences are ``T < T_c -> lambda > 0``, equality at the
    critical temperature, and ``T > T_c -> lambda < 0``.
    """
    temp = _finite_real_array(temperature, name="temperature")
    coefficient = _finite_real_array(a, name="a")
    critical = _finite_real_array(t_c, name="t_c")
    try:
        temp_b, coefficient_b, critical_b = np.broadcast_arrays(
            temp, coefficient, critical
        )
    except ValueError as exc:
        raise ValueError("temperature, a, and t_c are not broadcast-compatible") from exc
    result = coefficient_b * (critical_b - temp_b)
    return float(result) if result.ndim == 0 else result
