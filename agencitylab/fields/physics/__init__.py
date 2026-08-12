"""Research physics contract for future autonomous Agencity fields.

This isolated subpackage exposes the explicit ``beta_obs -> phi`` bridge,
Volume-2 quartic potential, U(1) vacuum primitives, and field-energy building
blocks. It is not re-exported from ``agencitylab.fields`` in v1.1.x.
"""

from agencitylab.scientific_status import ScientificStatus

from .bridge import beta_to_phi, phi_from_observable_field
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
    "SCIENTIFIC_STATUS",
    "QuarticAgencityPotential",
    "beta_to_phi",
    "dimensionless_benchmark",
    "field_energy_density",
    "gradient_energy_density",
    "kinetic_energy_density",
    "phi_from_observable_field",
    "potential_energy_density",
    "total_field_energy",
    "vacuum_amplitude",
    "vacuum_state",
]
