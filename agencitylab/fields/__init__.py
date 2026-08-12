"""Spatial and dynamical Agencity field interfaces.

Observable spatial fields remain ``experimental`` orchestration over the canonical
scalar pipeline. Autonomous ``phi`` physics, its classical dynamics, effective
beta-field models, conservation laws, and coherent structures are ``research``;
generic numerical operators remain ``experimental`` infrastructure.
"""

from agencitylab.models.field_result import ObservableAgencityFieldResult

from .coherent import (
    coherence_length,
    dimensionless_effective_potential,
    dimensionless_static_residual,
    domain_wall_profile,
    domain_wall_residual,
    field_zero_mask,
    from_dimensionless_field,
    phase_winding,
    to_dimensionless_field,
    vortex_field,
    vortex_radial_residual,
)
from .dynamics import (
    dissipative_klein_gordon_acceleration,
    klein_gordon_acceleration,
    simulate_dissipative_klein_gordon,
    simulate_klein_gordon,
    simulate_tdgl,
    tdgl_rhs,
)
from .effective_beta import (
    effective_beta_reaction,
    effective_beta_rhs,
    effective_beta_stationary_amplitude,
)
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
    FLAT_FIELD_METRIC_SIGNATURE,
    QuarticAgencityPotential,
    appendix_b_beta_energy_momentum_tensor,
    appendix_b_beta_equation_residual,
    appendix_b_beta_lagrangian_density,
    appendix_b_beta_noether_current,
    beta_to_phi,
    dimensionless_benchmark,
    field_energy_density,
    flat_energy_momentum_tensor,
    flat_field_lagrangian_density,
    flat_field_minkowski_metric,
    gradient_energy_density,
    kinetic_energy_density,
    phase_noether_current,
    phi_from_observable_field,
    potential_energy_density,
    radial_equation_residual,
    total_field_energy,
    u1_noether_current,
    vacuum_amplitude,
    vacuum_state,
)

# Compatibility alias for the historical observable-field name. New dynamical
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
    "FLAT_FIELD_METRIC_SIGNATURE",
    "flat_field_minkowski_metric",
    "flat_field_lagrangian_density",
    "flat_energy_momentum_tensor",
    "u1_noether_current",
    "phase_noether_current",
    "radial_equation_residual",
    "appendix_b_beta_lagrangian_density",
    "appendix_b_beta_equation_residual",
    "appendix_b_beta_noether_current",
    "appendix_b_beta_energy_momentum_tensor",
    "effective_beta_reaction",
    "effective_beta_rhs",
    "effective_beta_stationary_amplitude",
    "coherence_length",
    "to_dimensionless_field",
    "from_dimensionless_field",
    "dimensionless_effective_potential",
    "dimensionless_static_residual",
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
    "klein_gordon_acceleration",
    "dissipative_klein_gordon_acceleration",
    "tdgl_rhs",
    "simulate_klein_gordon",
    "simulate_dissipative_klein_gordon",
    "simulate_tdgl",
    "domain_wall_profile",
    "domain_wall_residual",
    "vortex_field",
    "vortex_radial_residual",
    "phase_winding",
    "field_zero_mask",
]
