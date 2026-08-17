"""Deterministic fixed-step simulators for classical Agencity field dynamics.

This integration layer connects the existing research Physics contract, generic
Numerics infrastructure, and shared dynamical-field result model.  It adds no
new potential, spatial stencil, boundary-condition type, or solution class.

Dirichlet states require an explicit integration-layer projection because the
existing spatial stencils impose fixed values only while evaluating operators,
whereas the generic time integrators intentionally perform no state projection.
Neumann conditions remain stencil-enforced only: this module does not claim
that an integrated state exactly preserves the prescribed derivative after a
finite time step.

Scientific status: research. No empirical validation is claimed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from agencitylab.models.field_extensions import (
    DynamicalAgencityFieldSolution,
    ParameterProvenance,
)
from agencitylab.scientific_status import ScientificStatus

from ..numerics import UniformRectilinearGrid, rk4_step, velocity_verlet_step
from ..numerics.boundaries import (
    Boundary,
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
    resolve_boundary,
)
from ..physics import QuarticAgencityPotential
from .dissipative import dissipative_klein_gordon_acceleration
from .klein_gordon import klein_gordon_acceleration
from .tdgl import tdgl_rhs

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH
FLAT_FIELD_METRIC_SIGNATURE = (1, -1, -1, -1)


def _validate_time_controls(dt: float, n_steps: int) -> tuple[float, int]:
    try:
        step = float(dt)
    except Exception as exc:
        raise ValueError("dt must be a finite real scalar") from exc
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    if isinstance(n_steps, (bool, np.bool_)):
        raise ValueError("n_steps must be an integer >= 1")
    try:
        steps = int(n_steps)
    except Exception as exc:
        raise ValueError("n_steps must be an integer >= 1") from exc
    if steps != n_steps or steps < 1:
        raise ValueError("n_steps must be an integer >= 1")
    return step, steps


def _validate_initial_field(
    value: np.ndarray,
    grid: UniformRectilinearGrid,
    *,
    name: str,
) -> np.ndarray:
    if not isinstance(grid, UniformRectilinearGrid):
        raise TypeError("grid must be a UniformRectilinearGrid")
    array = np.asarray(value)
    if array.shape != grid.shape:
        raise ValueError(f"{name} shape {array.shape} does not match grid shape {grid.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain real or complex numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    dtype = complex if np.iscomplexobj(array) else float
    return np.array(array, dtype=dtype, copy=True)


def _set_boundary_faces(field: np.ndarray, value: complex | float) -> np.ndarray:
    dtype = np.result_type(field.dtype, np.asarray(value).dtype)
    projected = np.array(field, dtype=dtype, copy=True)
    for axis in range(projected.ndim):
        lower = [slice(None)] * projected.ndim
        upper = [slice(None)] * projected.ndim
        lower[axis] = 0
        upper[axis] = -1
        projected[tuple(lower)] = value
        projected[tuple(upper)] = value
    return projected


def _project_field(
    field: np.ndarray,
    boundary: Boundary,
    *,
    derivative: bool = False,
) -> np.ndarray:
    if not isinstance(boundary, DirichletBoundary):
        return np.asarray(field)
    value: complex | float = 0.0 if derivative else boundary.value
    return _set_boundary_faces(np.asarray(field), value)


def _boundary_name(boundary: Boundary) -> str:
    if isinstance(boundary, PeriodicBoundary):
        return "periodic"
    if isinstance(boundary, DirichletBoundary):
        return "dirichlet"
    if isinstance(boundary, NeumannBoundary):
        return "neumann"
    raise TypeError("unsupported boundary type")


def _boundary_solver_metadata(boundary: Boundary) -> dict[str, Any]:
    if isinstance(boundary, DirichletBoundary):
        return {
            "boundary_handling": "stencil_plus_dirichlet_stage_and_state_projection",
            "state_boundary_projection": True,
        }
    if isinstance(boundary, NeumannBoundary):
        return {
            "boundary_handling": "stencil_only_no_state_projection",
            "state_boundary_projection": False,
        }
    return {
        "boundary_handling": "periodic_stencil",
        "state_boundary_projection": False,
    }


def _user_supplied(note: str) -> ParameterProvenance:
    return ParameterProvenance(source="user_supplied", note=note)


def _parameter_contract(
    potential: QuarticAgencityPotential,
    *,
    dt: float,
    n_steps: int,
    gamma: float | None = None,
) -> tuple[dict[str, Any], dict[str, ParameterProvenance]]:
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")

    parameters: dict[str, Any] = {
        "lambda": potential.lambda_,
        "mu": potential.mu,
        "dt": dt,
        "n_steps": n_steps,
    }
    provenance: dict[str, ParameterProvenance] = {
        "lambda": potential.lambda_provenance
        or _user_supplied("lambda supplied through QuarticAgencityPotential"),
        "mu": potential.mu_provenance
        or _user_supplied("mu supplied through QuarticAgencityPotential"),
        "dt": _user_supplied("fixed time step supplied to the simulator"),
        "n_steps": _user_supplied("integration step count supplied to the simulator"),
    }
    if gamma is not None:
        parameters["Gamma"] = gamma
        provenance["Gamma"] = _user_supplied(
            "friction coefficient supplied to the simulator"
        )
    return parameters, provenance


def _make_solution(
    *,
    times: np.ndarray,
    phi: np.ndarray,
    phi_dot: np.ndarray | None,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    boundary: Boundary,
    dynamics_name: str,
    integrator: str,
    integrator_order: int,
    dt: float,
    n_steps: int,
    gamma: float | None,
    metadata: Mapping[str, Any] | None,
) -> DynamicalAgencityFieldSolution:
    parameters, provenance = _parameter_contract(
        potential,
        dt=dt,
        n_steps=n_steps,
        gamma=gamma,
    )
    solver_metadata = {
        "integrator": integrator,
        "formal_time_order": integrator_order,
        "fixed_step": True,
        "metric_signature": FLAT_FIELD_METRIC_SIGNATURE,
        "metric_convention": "Chapter 16 flat field (+,-,-,-)",
        **_boundary_solver_metadata(boundary),
    }
    return DynamicalAgencityFieldSolution(
        times=times,
        phi=phi,
        phi_dot=phi_dot,
        spatial_shape=grid.shape,
        spatial_axes=grid.axes,
        metadata=dict(metadata or {}),
        parameters=parameters,
        parameter_provenance=provenance,
        dynamics_name=dynamics_name,
        boundary_name=_boundary_name(boundary),
        scientific_status=ScientificStatus.RESEARCH,
        solver_metadata=solver_metadata,
        units_convention=potential.units_convention,
    )


def simulate_klein_gordon(
    phi0: np.ndarray,
    phi_dot0: np.ndarray,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    *,
    dt: float,
    n_steps: int,
    boundary: Boundary | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DynamicalAgencityFieldSolution:
    """Simulate conservative Klein-Gordon dynamics with velocity-Verlet."""

    step, steps = _validate_time_controls(dt, n_steps)
    resolved = resolve_boundary(boundary)
    q = _validate_initial_field(phi0, grid, name="phi0")
    v = _validate_initial_field(phi_dot0, grid, name="phi_dot0")
    q = _project_field(q, resolved)
    v = _project_field(v, resolved, derivative=True)

    dtype = np.result_type(q.dtype, v.dtype)
    phi_history = np.empty((steps + 1, *grid.shape), dtype=dtype)
    velocity_history = np.empty_like(phi_history)
    phi_history[0] = q
    velocity_history[0] = v
    times = step * np.arange(steps + 1, dtype=float)

    def acceleration(_time: float, phi: np.ndarray, phi_dot: np.ndarray) -> np.ndarray:
        stage_phi = _project_field(phi, resolved)
        stage_velocity = _project_field(phi_dot, resolved, derivative=True)
        result = klein_gordon_acceleration(
            stage_phi,
            grid,
            potential,
            boundary=resolved,
        )
        return _project_field(result, resolved, derivative=True)

    for index in range(steps):
        q, v = velocity_verlet_step(
            acceleration,
            times[index],
            q,
            v,
            step,
        )
        q = _project_field(q, resolved)
        v = _project_field(v, resolved, derivative=True)
        phi_history[index + 1] = q
        velocity_history[index + 1] = v

    return _make_solution(
        times=times,
        phi=phi_history,
        phi_dot=velocity_history,
        grid=grid,
        potential=potential,
        boundary=resolved,
        dynamics_name="conservative_klein_gordon",
        integrator="velocity_verlet",
        integrator_order=2,
        dt=step,
        n_steps=steps,
        gamma=None,
        metadata=metadata,
    )


def simulate_dissipative_klein_gordon(
    phi0: np.ndarray,
    phi_dot0: np.ndarray,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    *,
    gamma: float,
    dt: float,
    n_steps: int,
    boundary: Boundary | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DynamicalAgencityFieldSolution:
    """Simulate dissipative Klein-Gordon dynamics with RK4.

    The velocity-dependent acceleration is written as a first-order system
    ``dphi/dt = pi`` and ``dpi/dt = acceleration(phi, pi)``.  Velocity-Verlet
    is intentionally not used for this dissipative equation.
    """

    step, steps = _validate_time_controls(dt, n_steps)
    resolved = resolve_boundary(boundary)
    q = _validate_initial_field(phi0, grid, name="phi0")
    v = _validate_initial_field(phi_dot0, grid, name="phi_dot0")
    q = _project_field(q, resolved)
    v = _project_field(v, resolved, derivative=True)

    # Validate gamma through the public equation primitive before allocating
    # the trajectory. This also rejects NaN/Inf consistently.
    dissipative_klein_gordon_acceleration(
        q,
        v,
        grid,
        potential,
        gamma,
        boundary=resolved,
    )
    gamma_value = float(gamma)

    state = np.stack((q, v), axis=0)
    phi_history = np.empty((steps + 1, *grid.shape), dtype=state.dtype)
    velocity_history = np.empty_like(phi_history)
    phi_history[0] = q
    velocity_history[0] = v
    times = step * np.arange(steps + 1, dtype=float)

    def rhs(_time: float, combined: np.ndarray) -> np.ndarray:
        stage_phi = _project_field(combined[0], resolved)
        stage_velocity = _project_field(
            combined[1],
            resolved,
            derivative=True,
        )
        dphi = _project_field(stage_velocity, resolved, derivative=True)
        dpi = dissipative_klein_gordon_acceleration(
            stage_phi,
            stage_velocity,
            grid,
            potential,
            gamma_value,
            boundary=resolved,
        )
        dpi = _project_field(dpi, resolved, derivative=True)
        return np.stack((dphi, dpi), axis=0)

    for index in range(steps):
        state = rk4_step(rhs, times[index], state, step)
        q = _project_field(state[0], resolved)
        v = _project_field(state[1], resolved, derivative=True)
        state = np.stack((q, v), axis=0)
        phi_history[index + 1] = q
        velocity_history[index + 1] = v

    return _make_solution(
        times=times,
        phi=phi_history,
        phi_dot=velocity_history,
        grid=grid,
        potential=potential,
        boundary=resolved,
        dynamics_name="dissipative_klein_gordon",
        integrator="rk4",
        integrator_order=4,
        dt=step,
        n_steps=steps,
        gamma=gamma_value,
        metadata=metadata,
    )


def simulate_tdgl(
    phi0: np.ndarray,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
    *,
    gamma: float,
    dt: float,
    n_steps: int,
    boundary: Boundary | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DynamicalAgencityFieldSolution:
    """Simulate overdamped TDGL dynamics with classical RK4.

    The returned solution leaves ``phi_dot`` absent.  TDGL is first order and
    storing a synthetic second-order velocity field would add no scientific
    information beyond the trajectory and equation parameters.
    """

    step, steps = _validate_time_controls(dt, n_steps)
    resolved = resolve_boundary(boundary)
    q = _validate_initial_field(phi0, grid, name="phi0")
    q = _project_field(q, resolved)

    # Validate gamma and the initial field through the public equation primitive.
    tdgl_rhs(q, grid, potential, gamma, boundary=resolved)
    gamma_value = float(gamma)

    phi_history = np.empty((steps + 1, *grid.shape), dtype=q.dtype)
    phi_history[0] = q
    times = step * np.arange(steps + 1, dtype=float)

    def rhs(_time: float, phi: np.ndarray) -> np.ndarray:
        stage_phi = _project_field(phi, resolved)
        result = tdgl_rhs(
            stage_phi,
            grid,
            potential,
            gamma_value,
            boundary=resolved,
        )
        return _project_field(result, resolved, derivative=True)

    for index in range(steps):
        q = rk4_step(rhs, times[index], q, step)
        q = _project_field(q, resolved)
        phi_history[index + 1] = q

    return _make_solution(
        times=times,
        phi=phi_history,
        phi_dot=None,
        grid=grid,
        potential=potential,
        boundary=resolved,
        dynamics_name="tdgl_overdamped",
        integrator="rk4",
        integrator_order=4,
        dt=step,
        n_steps=steps,
        gamma=gamma_value,
        metadata=metadata,
    )
