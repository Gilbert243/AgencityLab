"""Explicit research bridge from observable agencity ``beta`` to field ``phi``.

Scientific source: *Agencity — Advanced Mathematical Foundations and Extensions*,
Volume 2, Eq. (15.1): ``phi = sqrt(P_c * tau) * beta``.

This module has scientific status ``research``. It promotes an already-computed
observable agencity field into an autonomous field initialization in the user's
chosen dimensionless/natural-unit convention. It does not reconstruct a new
observable ``u``, does not modify the source observable result, and contains no
PDE solver or empirical validation claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agencitylab.models.field_result import ObservableAgencityFieldResult
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _normalise_time_axis(ndim: int, time_axis: int) -> int:
    if not isinstance(time_axis, (int, np.integer)):
        raise ValueError("time_axis must be an integer")
    axis = int(time_axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("time_axis is out of bounds for beta")
    return axis


def _finite_real_array(value: Any, *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} must be real numeric data") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def beta_to_phi(beta, P_c, tau, *, time_axis: int = 0) -> np.ndarray:
    """Promote observable ``beta`` to research field initialization ``phi``.

    The implemented Volume-2 relation is exactly
    ``phi = sqrt(P_c * tau) * beta``. No epsilon or inverse-power convention is
    introduced. Therefore local ``P_c == 0`` produces exact ``phi == 0`` even
    when ``beta`` is non-zero.

    Parameters
    ----------
    beta:
        Real or complex spacetime array containing one temporal axis and zero or
        more spatial axes.
    P_c:
        Characteristic power. Accepted forms are a finite non-negative scalar,
        an array with the exact spatial shape, or an array with the exact
        spacetime shape of ``beta``.
    tau:
        Characteristic structural time. Accepted forms are a finite strictly
        positive scalar or an array with the exact spatial shape. Spacetime
        ``tau(x,t)`` is intentionally not accepted in this version.
    time_axis:
        Temporal axis in ``beta``. Negative indexing is supported.

    Returns
    -------
    numpy.ndarray
        ``phi`` with exactly the same shape as ``beta``.

    Notes
    -----
    Scientific status: ``research``. Units are interpreted only in the
    dimensionless/natural-unit convention chosen by the caller. The autonomous
    ``phi`` model has no established empirical validation in AgencityLab.
    """
    raw_beta = np.asarray(beta)
    if raw_beta.ndim < 1:
        raise ValueError("beta must contain a temporal axis")
    if raw_beta.size == 0 or any(size == 0 for size in raw_beta.shape):
        raise ValueError("beta must be non-empty along every axis")
    if not np.all(np.isfinite(raw_beta)):
        raise ValueError("beta must contain only finite values")

    axis = _normalise_time_axis(raw_beta.ndim, time_axis)
    spatial_shape = raw_beta.shape[:axis] + raw_beta.shape[axis + 1 :]

    tau_arr = _finite_real_array(tau, name="tau")
    tau_resolved: float | np.ndarray
    if tau_arr.ndim == 0:
        tau_resolved = float(tau_arr)
        if tau_resolved <= 0.0:
            raise ValueError("tau must be strictly positive")
    elif tau_arr.shape == spatial_shape:
        if np.any(tau_arr <= 0.0):
            raise ValueError("tau must contain only strictly positive values")
        tau_resolved = tau_arr
    else:
        raise ValueError(
            f"tau must be a scalar or have exact spatial shape {spatial_shape}; "
            "spacetime tau(x,t) is not accepted"
        )

    power_arr = _finite_real_array(P_c, name="P_c")
    power_resolved: float | np.ndarray
    if power_arr.ndim == 0:
        power_resolved = float(power_arr)
        if power_resolved < 0.0:
            raise ValueError("P_c must be non-negative")
    elif power_arr.shape == spatial_shape:
        if np.any(power_arr < 0.0):
            raise ValueError("P_c must contain only non-negative values")
        power_resolved = power_arr
    elif power_arr.shape == raw_beta.shape:
        if np.any(power_arr < 0.0):
            raise ValueError("P_c must contain only non-negative values")
        power_resolved = power_arr
    else:
        raise ValueError(
            f"P_c must be a scalar, exact spatial shape {spatial_shape}, "
            f"or exact spacetime shape {raw_beta.shape}"
        )

    def expand_spatial(value):
        if np.ndim(value) == 0:
            return value
        return np.expand_dims(value, axis=axis)

    tau_scale = expand_spatial(tau_resolved)
    power_scale = (
        power_resolved
        if np.ndim(power_resolved) == raw_beta.ndim
        else expand_spatial(power_resolved)
    )
    return np.sqrt(power_scale * tau_scale) * raw_beta


def phi_from_observable_field(result: ObservableAgencityFieldResult) -> np.ndarray:
    """Explicitly promote an ``ObservableAgencityFieldResult`` to ``phi``.

    The source object is read only. Its experimental ``beta`` field is not
    altered, and the returned array is a separate research-layer field
    initialization.
    """
    if not isinstance(result, ObservableAgencityFieldResult):
        raise TypeError("result must be an ObservableAgencityFieldResult")
    return beta_to_phi(
        result.beta,
        result.P_c,
        result.tau,
        time_axis=result.time_axis,
    )
