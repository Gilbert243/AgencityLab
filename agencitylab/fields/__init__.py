"""Spatial and dynamical Agencity field interfaces.

Observable spatial fields remain ``experimental`` orchestration over the canonical
scalar pipeline.  Autonomous ``phi`` physics is ``research`` and generic numerical
operators are ``experimental`` infrastructure.  No PDE dynamics are implied by
importing this package.
"""

from agencitylab.models.field_result import ObservableAgencityFieldResult

from .local_field import compute_agencity_field
from .numerics import (
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
    UniformRectilinearGrid,
    diffusion_cfl_limit,
    gradient,
    gradient_norm_squared,
    integrate_spatial,
    laplacian,
    rk4_step,
    velocity_verlet_step,
    wave_cfl_limit,
)
from .physics import (
    QuarticAgencityPotential,
    beta_to_phi,
    dimensionless_benchmark,
    field_energy_density,
    gradient_energy_density,
    kinetic_energy_density,
    phi_from_observable_field,
    potential_energy_density,
    total_field_energy,
    vacuum_amplitude,
    vacuum_state,
)

# Compatibility alias for the historical observable-field name.  New dynamical
# field code must use DynamicalAgencityFieldState / Solution instead.
AgencityField = ObservableAgencityFieldResult

__all__ = [
    "AgencityField",
    "ObservableAgencityFieldResult",
    "compute_agencity_field",
    "beta_to_phi",
    "phi_from_observable_field",
    "QuarticAgencityPotential",
    "vacuum_amplitude",
    "vacuum_state",
    "dimensionless_benchmark",
    "kinetic_energy_density",
    "gradient_energy_density",
    "potential_energy_density",
    "field_energy_density",
    "total_field_energy",
    "UniformRectilinearGrid",
    "PeriodicBoundary",
    "DirichletBoundary",
    "NeumannBoundary",
    "gradient",
    "laplacian",
    "gradient_norm_squared",
    "integrate_spatial",
    "rk4_step",
    "velocity_verlet_step",
    "wave_cfl_limit",
    "diffusion_cfl_limit",
]
