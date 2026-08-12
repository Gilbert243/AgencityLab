"""Deterministic homogeneous flat-FLRW solver for speculative Agencity cosmology.

The solver integrates the Chapter-22 homogeneous field equation, scale-factor
equation, and acceleration Friedmann equation with the existing generic RK4
primitive. The first Friedmann equation is imposed only on the initial state
and retained as an explicit numerical constraint residual thereafter; it is
never silently projected or repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np

from agencitylab.fields.numerics import rk4_step
from agencitylab.fields.physics import QuarticAgencityPotential
from agencitylab.scientific_status import ScientificStatus

from .background import (
    field_acceleration,
    friedmann_constraint_residual,
    homogeneous_energy_density,
    homogeneous_pressure,
    hubble_derivative,
    initial_hubble_from_friedmann,
)

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


@dataclass(frozen=True, slots=True)
class FlatFLRWSolution:
    """Numerical solution of the homogeneous speculative FLRW field model."""

    times: np.ndarray
    phi: np.ndarray
    phi_dot: np.ndarray
    scale_factor: np.ndarray
    hubble: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray
    friedmann_residual: np.ndarray
    branch: str
    gravitational_constant: float
    scientific_status: ScientificStatus = ScientificStatus.SPECULATIVE
    model: str = "flat_flrw_agencity_field"
    units_convention: str = "natural_units"


def _finite_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_complex_scalar(value, *, name: str) -> complex:
    try:
        result = complex(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value, *, name: str) -> int:
    try:
        result = operator.index(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


def _real_scalar_array(value, *, name: str) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must evaluate to a scalar")
    arr = np.real_if_close(arr, tol=1000)
    if np.iscomplexobj(arr) or not np.isfinite(float(arr)):
        raise ValueError(f"{name} must be a finite theoretically real scalar")
    return float(arr)


def simulate_flat_flrw(
    *,
    phi0,
    phi_dot0,
    scale_factor0,
    potential: QuarticAgencityPotential,
    gravitational_constant,
    dt,
    steps,
    branch: str,
    t0=0.0,
) -> FlatFLRWSolution:
    """Integrate the homogeneous Chapter-22 Agencity cosmology equations.

    Parameters are explicit. ``branch`` selects the sign of the initial Hubble
    value through the first Friedmann equation. The subsequent Hubble evolution
    follows the acceleration equation, so a turnaround is not artificially
    forbidden. The first Friedmann relation is reported as a residual instead
    of being force-enforced after every numerical step.
    """
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    field0 = _finite_complex_scalar(phi0, name="phi0")
    velocity0 = _finite_complex_scalar(phi_dot0, name="phi_dot0")
    scale0 = _finite_scalar(scale_factor0, name="scale_factor0")
    if scale0 <= 0.0:
        raise ValueError("scale_factor0 must be strictly positive")
    gravitational = _finite_scalar(gravitational_constant, name="gravitational_constant")
    if gravitational <= 0.0:
        raise ValueError("gravitational_constant must be strictly positive")
    step = _finite_scalar(dt, name="dt")
    if step <= 0.0:
        raise ValueError("dt must be strictly positive")
    count = _positive_integer(steps, name="steps")
    start_time = _finite_scalar(t0, name="t0")

    rho0 = _real_scalar_array(
        homogeneous_energy_density(field0, velocity0, potential),
        name="initial rho",
    )
    hubble0 = initial_hubble_from_friedmann(
        rho0,
        gravitational_constant=gravitational,
        branch=branch,
    )

    times = start_time + step * np.arange(count + 1, dtype=float)
    states = np.empty((count + 1, 6), dtype=float)
    states[0] = np.array(
        [field0.real, field0.imag, velocity0.real, velocity0.imag, scale0, hubble0],
        dtype=float,
    )

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        phi = complex(state[0], state[1])
        phi_dot = complex(state[2], state[3])
        scale_factor = float(state[4])
        hubble = float(state[5])
        if scale_factor <= 0.0:
            raise ValueError("scale factor became non-positive during integration")

        rho = _real_scalar_array(
            homogeneous_energy_density(phi, phi_dot, potential),
            name="rho",
        )
        pressure = _real_scalar_array(
            homogeneous_pressure(phi, phi_dot, potential),
            name="pressure",
        )
        phi_ddot = complex(np.asarray(field_acceleration(phi, phi_dot, hubble, potential)).item())
        hubble_dot = _real_scalar_array(
            hubble_derivative(
                hubble,
                rho,
                pressure,
                gravitational_constant=gravitational,
            ),
            name="hubble_dot",
        )
        return np.array(
            [
                phi_dot.real,
                phi_dot.imag,
                phi_ddot.real,
                phi_ddot.imag,
                hubble * scale_factor,
                hubble_dot,
            ],
            dtype=float,
        )

    for index in range(count):
        states[index + 1] = rk4_step(rhs, times[index], states[index], step)
        if not np.all(np.isfinite(states[index + 1])):
            raise ValueError("FLRW integration produced NaN or Inf")
        if states[index + 1, 4] <= 0.0:
            raise ValueError("scale factor became non-positive during integration")

    phi = states[:, 0] + 1j * states[:, 1]
    phi_dot = states[:, 2] + 1j * states[:, 3]
    scale_factor = states[:, 4].copy()
    hubble = states[:, 5].copy()
    rho = np.asarray(homogeneous_energy_density(phi, phi_dot, potential), dtype=float)
    pressure = np.asarray(homogeneous_pressure(phi, phi_dot, potential), dtype=float)
    friedmann_residual = np.array(
        [
            float(
                np.asarray(
                    friedmann_constraint_residual(
                        current_hubble,
                        current_rho,
                        gravitational_constant=gravitational,
                    )
                )
            )
            for current_hubble, current_rho in zip(hubble, rho, strict=True)
        ],
        dtype=float,
    )

    return FlatFLRWSolution(
        times=times,
        phi=phi,
        phi_dot=phi_dot,
        scale_factor=scale_factor,
        hubble=hubble,
        rho=rho,
        pressure=pressure,
        friedmann_residual=friedmann_residual,
        branch=branch,
        gravitational_constant=gravitational,
    )
