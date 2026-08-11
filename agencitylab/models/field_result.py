"""Result model for experimental observable spatial Agencity fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ObservableAgencityFieldResult:
    """Pointwise spatial orchestration of the canonical scalar Agencity pipeline.

    The scalar quantities stored at every spatial location are canonical outputs
    of :func:`agencitylab.compute_agencity`.  The spatial orchestration itself is
    experimental and does not define an autonomous dynamical field.
    """

    t: np.ndarray
    spatial_axes: tuple[np.ndarray, ...]
    u: np.ndarray
    u_star: np.ndarray
    X_star: np.ndarray
    A_star: np.ndarray
    M: np.ndarray
    O: np.ndarray
    D: np.ndarray
    S: np.ndarray
    J: np.ndarray
    U: np.ndarray
    beta: np.ndarray
    b: np.ndarray
    A_ref: np.ndarray
    tau: np.ndarray
    w: np.ndarray
    P_c: np.ndarray
    time_axis: int
    spatial_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "experimental"
    model: str = "observable_agencity_field"
    backend: str = "numpy"

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        if self.t.ndim != 1 or self.t.size < 3:
            raise ValueError("t must be one-dimensional with at least three samples")
        if not np.all(np.isfinite(self.t)) or np.any(np.diff(self.t) <= 0.0):
            raise ValueError("t must contain finite strictly increasing samples")

        self.u = np.asarray(self.u, dtype=float)
        shape = self.u.shape
        if self.u.ndim < 2:
            raise ValueError("observable field must contain time and at least one spatial axis")
        if not np.all(np.isfinite(self.u)):
            raise ValueError("u must contain only finite values")
        if not 0 <= self.time_axis < self.u.ndim:
            raise ValueError("time_axis is inconsistent with u")
        if shape[self.time_axis] != self.t.size:
            raise ValueError("t length must match the temporal dimension of u")

        expected_spatial_shape = shape[: self.time_axis] + shape[self.time_axis + 1 :]
        if tuple(self.spatial_shape) != tuple(expected_spatial_shape):
            raise ValueError("spatial_shape is inconsistent with u and time_axis")
        self.spatial_shape = tuple(int(size) for size in self.spatial_shape)

        if len(self.spatial_axes) != len(self.spatial_shape):
            raise ValueError("spatial_axes must contain one coordinate per spatial dimension")
        axes: list[np.ndarray] = []
        for index, (axis, size) in enumerate(zip(self.spatial_axes, self.spatial_shape)):
            arr = np.asarray(axis, dtype=float)
            if arr.ndim != 1 or arr.size != size:
                raise ValueError(f"spatial_axes[{index}] must be one-dimensional with length {size}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"spatial_axes[{index}] must contain only finite values")
            if arr.size > 1:
                delta = np.diff(arr)
                if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
                    raise ValueError(f"spatial_axes[{index}] must be strictly monotone")
            axes.append(arr)
        self.spatial_axes = tuple(axes)

        real_fields = ("u_star", "X_star", "A_star", "M", "O", "D", "S", "J")
        complex_fields = ("U", "beta", "b")
        for name in real_fields:
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite and have shape {shape}")
            setattr(self, name, arr)
        for name in complex_fields:
            arr = np.asarray(getattr(self, name), dtype=complex)
            if arr.shape != shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite and have shape {shape}")
            setattr(self, name, arr)

        for name in ("A_ref", "tau", "w"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != self.spatial_shape:
                raise ValueError(f"{name} must have resolved spatial shape {self.spatial_shape}")
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError(f"{name} must contain strictly positive finite values")
            setattr(self, name, arr)

        self.P_c = np.asarray(self.P_c, dtype=float)
        if self.P_c.shape != shape:
            raise ValueError("P_c must have the resolved spacetime shape of u")
        if not np.all(np.isfinite(self.P_c)) or np.any(self.P_c < 0.0):
            raise ValueError("P_c must contain only non-negative finite values")

        self.metadata = dict(self.metadata or {})
        if self.status != "experimental":
            raise ValueError("ObservableAgencityFieldResult status must be 'experimental'")
        if self.model != "observable_agencity_field":
            raise ValueError("unexpected observable-field model identifier")
        if self.backend != "numpy":
            raise ValueError("v1.1 observable fields use the NumPy reference backend")

    @property
    def beta_obs(self) -> np.ndarray:
        """Explicit observable-field spelling for the intrinsic Agencity state."""
        return self.beta

    @property
    def b_obs(self) -> np.ndarray:
        """Explicit observable-field spelling for the Agencity flux."""
        return self.b
