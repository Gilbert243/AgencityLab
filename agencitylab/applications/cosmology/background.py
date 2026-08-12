"""Homogeneous flat-FLRW primitives for speculative Agencity cosmology.

Scientific source: *Agencity — Advanced Mathematical Foundations and Extensions*,
Volume 2, Chapters 20 and 22. The cosmological extension is explicitly
speculative. These functions do not alter the canonical observable Agencity
pipeline and do not claim that the quartic field explains inflation or dark
energy.
"""

from __future__ import annotations

import numpy as np

from agencitylab.fields.physics import (
    QuarticAgencityPotential,
    field_energy_density,
    kinetic_energy_density,
    potential_energy_density,
)
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


def _finite_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_gravitational_constant(value) -> float:
    result = _finite_scalar(value, name="gravitational_constant")
    if result <= 0.0:
        raise ValueError("gravitational_constant must be strictly positive")
    return result


def _potential(value) -> QuarticAgencityPotential:
    if not isinstance(value, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    return value


def homogeneous_energy_density(phi, phi_dot, potential: QuarticAgencityPotential) -> np.ndarray:
    """Return ``rho_phi = 1/2 |phi_dot|^2 + V(|phi|)``.

    The implementation reuses the shared field-energy primitive with exactly
    zero spatial-gradient contribution for a homogeneous FLRW field.
    """
    potential = _potential(potential)
    return field_energy_density(phi, phi_dot, 0.0, potential)


def homogeneous_pressure(phi, phi_dot, potential: QuarticAgencityPotential) -> np.ndarray:
    """Return ``p_phi = 1/2 |phi_dot|^2 - V(|phi|)``."""
    potential = _potential(potential)
    kinetic = kinetic_energy_density(phi_dot)
    potential_density = potential_energy_density(phi, potential)
    try:
        shape = np.broadcast_shapes(kinetic.shape, potential_density.shape)
    except ValueError as exc:
        raise ValueError("phi and phi_dot have incompatible shapes") from exc
    return np.broadcast_to(kinetic, shape) - np.broadcast_to(potential_density, shape)


def equation_of_state_parameter(rho, pressure) -> np.ndarray:
    """Return ``w_phi = p_phi / rho_phi`` where ``rho_phi`` is non-zero.

    No epsilon is inserted when ``rho_phi = 0`` because the equation-of-state
    parameter is then undefined rather than regularised physics.
    """
    rho_array = np.asarray(rho, dtype=float)
    pressure_array = np.asarray(pressure, dtype=float)
    if not np.all(np.isfinite(rho_array)) or not np.all(np.isfinite(pressure_array)):
        raise ValueError("rho and pressure must contain only finite real values")
    try:
        shape = np.broadcast_shapes(rho_array.shape, pressure_array.shape)
    except ValueError as exc:
        raise ValueError("rho and pressure have incompatible shapes") from exc
    rho_array = np.broadcast_to(rho_array, shape)
    pressure_array = np.broadcast_to(pressure_array, shape)
    if np.any(rho_array == 0.0):
        raise ValueError("equation-of-state parameter is undefined where rho == 0")
    return pressure_array / rho_array


def field_acceleration(
    phi,
    phi_dot,
    hubble,
    potential: QuarticAgencityPotential,
) -> np.ndarray:
    """Return the homogeneous field acceleration ``phi_ddot``.

    The Chapter-22 equation is
    ``phi_ddot + 3 H phi_dot + V'(|phi|) phi/|phi| = 0``.
    The shared ``potential.gradient(phi)`` is the already-audited equivalent
    field term and remains well defined at ``phi = 0`` without physical EPS.
    """
    potential = _potential(potential)
    field = np.asarray(phi)
    velocity = np.asarray(phi_dot)
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(velocity)):
        raise ValueError("phi and phi_dot must contain only finite values")
    hubble_value = _finite_scalar(hubble, name="hubble")
    try:
        shape = np.broadcast_shapes(field.shape, velocity.shape)
    except ValueError as exc:
        raise ValueError("phi and phi_dot have incompatible shapes") from exc
    field = np.broadcast_to(field, shape)
    velocity = np.broadcast_to(velocity, shape)
    return -3.0 * hubble_value * velocity - potential.gradient(field)


def friedmann_constraint_residual(hubble, rho, *, gravitational_constant) -> np.ndarray:
    """Return ``H^2 - (8 pi G / 3) rho_phi``."""
    hubble_value = _finite_scalar(hubble, name="hubble")
    gravitational = _positive_gravitational_constant(gravitational_constant)
    density = np.asarray(rho, dtype=float)
    if not np.all(np.isfinite(density)):
        raise ValueError("rho must contain only finite real values")
    return hubble_value**2 - (8.0 * np.pi * gravitational / 3.0) * density


def acceleration_equation_residual(
    hubble_dot,
    hubble,
    rho,
    pressure,
    *,
    gravitational_constant,
) -> np.ndarray:
    """Return the Chapter-22 acceleration-equation residual.

    The residual is
    ``H_dot + H^2 + (4 pi G / 3) (rho_phi + 3 p_phi)``.
    """
    hubble_dot_value = _finite_scalar(hubble_dot, name="hubble_dot")
    hubble_value = _finite_scalar(hubble, name="hubble")
    gravitational = _positive_gravitational_constant(gravitational_constant)
    density = np.asarray(rho, dtype=float)
    pressure_array = np.asarray(pressure, dtype=float)
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(pressure_array)):
        raise ValueError("rho and pressure must contain only finite real values")
    try:
        shape = np.broadcast_shapes(density.shape, pressure_array.shape)
    except ValueError as exc:
        raise ValueError("rho and pressure have incompatible shapes") from exc
    density = np.broadcast_to(density, shape)
    pressure_array = np.broadcast_to(pressure_array, shape)
    return (
        hubble_dot_value
        + hubble_value**2
        + (4.0 * np.pi * gravitational / 3.0) * (density + 3.0 * pressure_array)
    )


def hubble_derivative(
    hubble,
    rho,
    pressure,
    *,
    gravitational_constant,
) -> np.ndarray:
    """Return ``H_dot`` from the Chapter-22 acceleration equation."""
    hubble_value = _finite_scalar(hubble, name="hubble")
    gravitational = _positive_gravitational_constant(gravitational_constant)
    density = np.asarray(rho, dtype=float)
    pressure_array = np.asarray(pressure, dtype=float)
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(pressure_array)):
        raise ValueError("rho and pressure must contain only finite real values")
    try:
        shape = np.broadcast_shapes(density.shape, pressure_array.shape)
    except ValueError as exc:
        raise ValueError("rho and pressure have incompatible shapes") from exc
    density = np.broadcast_to(density, shape)
    pressure_array = np.broadcast_to(pressure_array, shape)
    return -hubble_value**2 - (4.0 * np.pi * gravitational / 3.0) * (
        density + 3.0 * pressure_array
    )


def initial_hubble_from_friedmann(
    rho,
    *,
    gravitational_constant,
    branch: str,
) -> float:
    """Return the real initial ``H`` from the first Friedmann equation.

    ``branch`` must be explicitly ``"expanding"`` or ``"contracting"``.
    Negative ``rho`` is rejected because no real flat-FLRW Hubble value can
    satisfy ``H^2 = (8 pi G / 3) rho`` in that case.
    """
    density = _finite_scalar(rho, name="rho")
    gravitational = _positive_gravitational_constant(gravitational_constant)
    if density < 0.0:
        raise ValueError("negative rho has no real flat-FLRW Hubble solution")
    if branch not in {"expanding", "contracting"}:
        raise ValueError("branch must be 'expanding' or 'contracting'")
    magnitude = float(np.sqrt((8.0 * np.pi * gravitational / 3.0) * density))
    return magnitude if branch == "expanding" else -magnitude
