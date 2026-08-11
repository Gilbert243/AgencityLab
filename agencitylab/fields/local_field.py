"""Experimental observable spatial-field orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from agencitylab.api.compute import compute_agencity
from agencitylab.exceptions import AgencityValidationError, PhysicalParameterError
from agencitylab.models.field_result import ObservableAgencityFieldResult
from agencitylab.version import __version__


def _normalise_time_axis(ndim: int, time_axis: int) -> int:
    if not isinstance(time_axis, (int, np.integer)):
        raise AgencityValidationError("time_axis must be an integer")
    axis = int(time_axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise AgencityValidationError("time_axis is out of bounds for u")
    return axis


def _validate_time(t: Any, expected_length: int) -> np.ndarray:
    try:
        values = np.asarray(t, dtype=float)
    except Exception as exc:
        raise AgencityValidationError("t must be numeric") from exc
    if values.ndim != 1:
        raise AgencityValidationError("t must be one-dimensional")
    if values.size < 3:
        raise AgencityValidationError("temporal axis must contain at least three samples")
    if values.size != expected_length:
        raise AgencityValidationError("t length must match u along time_axis")
    if not np.all(np.isfinite(values)):
        raise AgencityValidationError("t must contain only finite values")
    if np.any(np.diff(values) <= 0.0):
        raise AgencityValidationError("t must be strictly increasing")
    return values


def _resolve_spatial_parameter(value: Any, spatial_shape: tuple[int, ...], name: str):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise PhysicalParameterError(f"{name} must be numeric") from exc
    if arr.ndim == 0:
        scalar = float(arr)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise PhysicalParameterError(f"{name} must be strictly positive and finite")
        return np.full(spatial_shape, scalar, dtype=float), "scalar"
    if arr.shape != spatial_shape:
        raise PhysicalParameterError(
            f"{name} must be a scalar or have exact spatial shape {spatial_shape}; "
            "time-dependent values are not accepted in v1.1"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise PhysicalParameterError(f"{name} must contain strictly positive finite values")
    return arr.astype(float, copy=False), "spatial"


def _resolve_power(
    value: Any,
    original_shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
    time_axis: int,
    n_time: int,
):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise PhysicalParameterError("P_c must be numeric") from exc
    if arr.ndim == 0:
        scalar = float(arr)
        if not np.isfinite(scalar) or scalar < 0.0:
            raise PhysicalParameterError("P_c must be non-negative and finite")
        time_first = np.full((n_time, *spatial_shape), scalar, dtype=float)
        return time_first, "scalar"
    if arr.shape == spatial_shape:
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise PhysicalParameterError("P_c must contain only non-negative finite values")
        time_first = np.broadcast_to(arr, (n_time, *spatial_shape)).astype(float, copy=True)
        return time_first, "spatial"
    if arr.shape == original_shape:
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise PhysicalParameterError("P_c must contain only non-negative finite values")
        return np.moveaxis(arr, time_axis, 0).astype(float, copy=False), "spacetime"
    raise PhysicalParameterError(
        f"P_c must be a scalar, exact spatial shape {spatial_shape}, "
        f"or exact spacetime shape {original_shape}"
    )


def _resolve_spatial_axes(spatial_axes: Any, spatial_shape: tuple[int, ...]):
    if spatial_axes is None:
        return tuple(np.arange(size, dtype=float) for size in spatial_shape), "sample_index"
    if not isinstance(spatial_axes, (tuple, list)):
        raise AgencityValidationError("spatial_axes must be a tuple/list of one-dimensional axes")
    if len(spatial_axes) != len(spatial_shape):
        raise AgencityValidationError(
            f"spatial_axes must contain {len(spatial_shape)} axes for spatial shape {spatial_shape}"
        )
    resolved = []
    for index, (axis, size) in enumerate(zip(spatial_axes, spatial_shape)):
        try:
            arr = np.asarray(axis, dtype=float)
        except Exception as exc:
            raise AgencityValidationError(f"spatial_axes[{index}] must be numeric") from exc
        if arr.ndim != 1 or arr.size != size:
            raise AgencityValidationError(
                f"spatial_axes[{index}] must be one-dimensional with length {size}"
            )
        if not np.all(np.isfinite(arr)):
            raise AgencityValidationError(f"spatial_axes[{index}] must contain only finite values")
        if arr.size > 1:
            delta = np.diff(arr)
            if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
                raise AgencityValidationError(f"spatial_axes[{index}] must be strictly monotone")
        resolved.append(arr)
    return tuple(resolved), "explicit"


def _user_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if hasattr(metadata, "to_dict") and callable(metadata.to_dict):
        metadata = metadata.to_dict()
    if not isinstance(metadata, dict):
        raise AgencityValidationError("metadata must be a dictionary-like object or None")
    return dict(metadata)


def compute_agencity_field(
    u,
    t,
    *,
    spatial_axes=None,
    A_ref,
    tau,
    w=None,
    P_c,
    time_axis: int = 0,
    metadata=None,
) -> ObservableAgencityFieldResult:
    """Compute an experimental observable spatial Agencity field.

    Each fixed spatial position is treated as an independent temporal observable
    and passed through the reference scalar :func:`compute_agencity` pipeline.
    No spatial CRM, spatial derivative, PDE, smoothing, or autonomous ``phi``
    dynamics is introduced here.
    """
    try:
        field = np.asarray(u, dtype=float)
    except Exception as exc:
        raise AgencityValidationError("u must be a numeric array") from exc
    if field.ndim < 2:
        raise AgencityValidationError(
            "u must contain one temporal axis and at least one spatial axis"
        )
    if field.size == 0 or any(size == 0 for size in field.shape):
        raise AgencityValidationError("u must be non-empty along every axis")
    if not np.all(np.isfinite(field)):
        raise AgencityValidationError("u must contain only finite values")

    axis = _normalise_time_axis(field.ndim, time_axis)
    times = _validate_time(t, field.shape[axis])
    spatial_shape = field.shape[:axis] + field.shape[axis + 1 :]
    axes, spatial_axes_mode = _resolve_spatial_axes(spatial_axes, spatial_shape)

    A_ref_resolved, A_ref_mode = _resolve_spatial_parameter(A_ref, spatial_shape, "A_ref")
    tau_resolved, tau_mode = _resolve_spatial_parameter(tau, spatial_shape, "tau")
    if w is None:
        w_resolved = tau_resolved.copy()
        w_mode = "fallback_w_equals_tau"
    else:
        w_resolved, supplied_w_mode = _resolve_spatial_parameter(w, spatial_shape, "w")
        w_mode = f"explicit_{supplied_w_mode}"

    time_first = np.moveaxis(field, axis, 0)
    power_time_first, P_c_mode = _resolve_power(
        P_c, field.shape, spatial_shape, axis, times.size
    )

    n_points = int(np.prod(spatial_shape, dtype=int))
    u_flat = time_first.reshape(times.size, n_points)
    power_flat = power_time_first.reshape(times.size, n_points)
    A_ref_flat = A_ref_resolved.reshape(n_points)
    tau_flat = tau_resolved.reshape(n_points)
    w_flat = w_resolved.reshape(n_points)

    real_names = ("u_star", "X_star", "A_star", "M", "O", "D", "S", "J")
    complex_names = ("U", "beta", "b")
    real_outputs = {name: np.empty_like(u_flat, dtype=float) for name in real_names}
    complex_outputs = {
        name: np.empty(u_flat.shape, dtype=complex) for name in complex_names
    }

    for point in range(n_points):
        scalar = compute_agencity(
            u=u_flat[:, point],
            xi=times,
            A_ref=float(A_ref_flat[point]),
            tau=float(tau_flat[point]),
            w=float(w_flat[point]),
            P_c=power_flat[:, point],
            config={"backend": "numpy"},
        )
        for name in real_names:
            real_outputs[name][:, point] = getattr(scalar, name)
        for name in complex_names:
            complex_outputs[name][:, point] = getattr(scalar, name)

    def restore(values: np.ndarray) -> np.ndarray:
        return np.moveaxis(values.reshape((times.size, *spatial_shape)), 0, axis)

    resolved_metadata = {
        "agencitylab_version": __version__,
        "status": "experimental",
        "model": "observable_agencity_field",
        "backend": "numpy",
        "field_shape": tuple(int(size) for size in field.shape),
        "time_axis": axis,
        "spatial_shape": tuple(int(size) for size in spatial_shape),
        "spatial_axes_mode": spatial_axes_mode,
        "A_ref_mode": A_ref_mode,
        "tau_mode": tau_mode,
        "w_mode": w_mode,
        "w_resolution": (
            "w was unspecified; implementation convention w = tau was used"
            if w is None
            else "w was supplied explicitly and preserved independently of tau"
        ),
        "P_c_mode": P_c_mode,
        "crm_scope": "temporal_only_independent_at_each_spatial_location",
        "scientific_scope": (
            "experimental spatial orchestration over the canonical scalar pipeline"
        ),
        "user_metadata": _user_metadata(metadata),
    }

    return ObservableAgencityFieldResult(
        t=times,
        spatial_axes=axes,
        u=field,
        u_star=restore(real_outputs["u_star"]),
        X_star=restore(real_outputs["X_star"]),
        A_star=restore(real_outputs["A_star"]),
        M=restore(real_outputs["M"]),
        O=restore(real_outputs["O"]),
        D=restore(real_outputs["D"]),
        S=restore(real_outputs["S"]),
        J=restore(real_outputs["J"]),
        U=restore(complex_outputs["U"]),
        beta=restore(complex_outputs["beta"]),
        b=restore(complex_outputs["b"]),
        A_ref=A_ref_resolved,
        tau=tau_resolved,
        w=w_resolved,
        P_c=np.moveaxis(power_time_first, 0, axis),
        time_axis=axis,
        spatial_shape=spatial_shape,
        metadata=resolved_metadata,
    )
