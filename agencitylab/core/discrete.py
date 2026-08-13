"""Explicit discrete derivative operators from Volume 2 of the Theory of Agencity.

These operators are distinct from the successive ``numpy.gradient`` numerical
approximation used to sample the continuous reference pipeline. For interior
samples they implement exactly

``X_n = (u[n+1] - u[n-1]) / (2 delta)``

and

``A_n = (u[n+1] - 2 u[n] + u[n-1]) / delta**2``.

To preserve result length, boundaries use explicit second-order one-sided
finite differences. For the minimal three-sample case the only available
three-point second difference is reused at both endpoints. These boundary rules
are implementation conventions; they do not change the interior Volume-2
stencils.
"""

from __future__ import annotations

import numpy as np

from .validation import validate_positive_scalar, validate_signal


def _discrete_sequence(values):
    values = validate_signal(values, name="values")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return values


def volume2_first_difference(values, delta):
    """Return the Volume-2 centred first difference with one-sided boundaries."""
    values = _discrete_sequence(values)
    delta = validate_positive_scalar(delta, name="delta")
    out = np.empty_like(values, dtype=float)

    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * delta)
    out[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * delta)
    out[-1] = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * delta)
    return out


def volume2_second_difference(values, delta):
    """Return the Volume-2 centred second difference with explicit boundaries."""
    values = _discrete_sequence(values)
    delta = validate_positive_scalar(delta, name="delta")
    out = np.empty_like(values, dtype=float)
    delta2 = delta * delta

    out[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / delta2
    if values.size >= 4:
        out[0] = (
            2.0 * values[0] - 5.0 * values[1] + 4.0 * values[2] - values[3]
        ) / delta2
        out[-1] = (
            2.0 * values[-1]
            - 5.0 * values[-2]
            + 4.0 * values[-3]
            - values[-4]
        ) / delta2
    else:
        boundary = (values[2] - 2.0 * values[1] + values[0]) / delta2
        out[0] = boundary
        out[-1] = boundary
    return out


__all__ = ["volume2_first_difference", "volume2_second_difference"]
