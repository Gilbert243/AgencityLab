"""Generic deterministic time-integration primitives for NumPy states.

The integrators in this module contain no Agencity-specific equations and do
not apply spatial boundary conditions implicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Array = np.ndarray


def _validate_dt(dt: float) -> float:
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    return value


def _validate_state(state: Array, name: str) -> Array:
    array = np.asarray(state)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain numeric real or complex values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _evaluate_like(
    callback: Callable[..., Array],
    expected_shape: tuple[int, ...],
    *args,
    name: str,
) -> Array:
    value = np.asarray(callback(*args))
    if value.shape != expected_shape:
        raise ValueError(
            f"{name} callback returned shape {value.shape}; expected {expected_shape}"
        )
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        raise TypeError(f"{name} callback must return numeric values")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} callback returned NaN or Inf")
    return value


def rk4_step(
    rhs: Callable[[float, Array], Array],
    t: float,
    state: Array,
    dt: float,
) -> Array:
    """Advance ``dstate/dt = rhs(t, state)`` by one classical RK4 step.

    The method is explicit, deterministic, fourth-order accurate for smooth
    right-hand sides, and supports arbitrary real or complex NumPy array
    shapes.  The callback must return exactly the same shape as ``state``.
    """

    step = _validate_dt(dt)
    y = _validate_state(state, "state")
    time = float(t)
    if not np.isfinite(time):
        raise ValueError("t must be finite")

    k1 = _evaluate_like(rhs, y.shape, time, y, name="rhs")
    k2 = _evaluate_like(rhs, y.shape, time + 0.5 * step, y + 0.5 * step * k1, name="rhs")
    k3 = _evaluate_like(rhs, y.shape, time + 0.5 * step, y + 0.5 * step * k2, name="rhs")
    k4 = _evaluate_like(rhs, y.shape, time + step, y + step * k3, name="rhs")
    return y + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def velocity_verlet_step(
    acceleration: Callable[[float, Array, Array], Array],
    t: float,
    phi: Array,
    phi_dot: Array,
    dt: float,
) -> tuple[Array, Array]:
    """Advance a generic second-order system by an explicit Verlet predictor.

    Solves ``phi_ddot = acceleration(t, phi, phi_dot)`` using a position-Verlet
    update and a predicted velocity for the endpoint acceleration.  For smooth
    systems this is second-order accurate; it reduces to the usual
    velocity-Verlet method when acceleration is velocity independent.  The
    method supports real or complex N-D NumPy arrays and applies no hidden
    boundary projection.
    """

    step = _validate_dt(dt)
    q = _validate_state(phi, "phi")
    v = _validate_state(phi_dot, "phi_dot")
    if q.shape != v.shape:
        raise ValueError("phi and phi_dot must have identical shapes")
    time = float(t)
    if not np.isfinite(time):
        raise ValueError("t must be finite")

    a0 = _evaluate_like(acceleration, q.shape, time, q, v, name="acceleration")
    q_new = q + step * v + 0.5 * step * step * a0
    v_predict = v + step * a0
    a1 = _evaluate_like(
        acceleration,
        q.shape,
        time + step,
        q_new,
        v_predict,
        name="acceleration",
    )
    v_new = v + 0.5 * step * (a0 + a1)
    return q_new, v_new
