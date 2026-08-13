"""Research physics contract for autonomous and field-theoretic Agencity models.

This subpackage exposes the explicit ``beta_obs -> phi`` bridge, Volume-2
quartic potential, U(1) vacuum primitives, field-energy building blocks,
Chapter-16 flat-field conservation laws, and the separately documented
Appendix-B beta-field formulation.
"""

from agencitylab.scientific_status import ScientificStatus

from .beta_formulation import (
    appendix_b_beta_energy_momentum_tensor,
    appendix_b_beta_equation_residual,
    appendix_b_beta_lagrangian_density,
    appendix_b_beta_noether_current,
)
from .bridge import beta_to_phi, phi_from_observable_field
from .conservation import (
    FLAT_FIELD_METRIC_SIGNATURE,
    flat_energy_momentum_tensor,
    flat_field_lagrangian_density,
    flat_field_minkowski_metric,
    phase_noether_current,
    radial_equation_residual,
    u1_noether_current,
)
from .energy import (
    field_energy_density,
    gradient_energy_density,
    kinetic_energy_density,
    potential_energy_density,
    total_field_energy,
)
from .potential import QuarticAgencityPotential, vacuum_amplitude, vacuum_state
from .presets import dimensionless_benchmark

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "FLAT_FIELD_METRIC_SIGNATURE",
    "SCIENTIFIC_STATUS",
    "QuarticAgencityPotential",
    "appendix_b_beta_energy_momentum_tensor",
    "appendix_b_beta_equation_residual",
    "appendix_b_beta_lagrangian_density",
    "appendix_b_beta_noether_current",
    "beta_to_phi",
    "dimensionless_benchmark",
    "field_energy_density",
    "flat_energy_momentum_tensor",
    "flat_field_lagrangian_density",
    "flat_field_minkowski_metric",
    "gradient_energy_density",
    "kinetic_energy_density",
    "phase_noether_current",
    "phi_from_observable_field",
    "potential_energy_density",
    "radial_equation_residual",
    "total_field_energy",
    "u1_noether_current",
    "vacuum_amplitude",
    "vacuum_state",
]
