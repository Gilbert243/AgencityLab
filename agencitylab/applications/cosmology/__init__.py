"""Speculative FLRW cosmology for the autonomous Agencity field.

The package implements only the homogeneous flat-FLRW equations explicitly
stated in Volume 2. It does not claim observational validation, inflation, or a
dark-energy explanation.
"""

from agencitylab.scientific_status import ScientificStatus

from .background import (
    acceleration_equation_residual,
    equation_of_state_parameter,
    field_acceleration,
    friedmann_constraint_residual,
    homogeneous_energy_density,
    homogeneous_pressure,
    hubble_derivative,
    initial_hubble_from_friedmann,
)
from .solver import FlatFLRWSolution, simulate_flat_flrw

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE

__all__ = [
    "SCIENTIFIC_STATUS",
    "FlatFLRWSolution",
    "acceleration_equation_residual",
    "equation_of_state_parameter",
    "field_acceleration",
    "friedmann_constraint_residual",
    "homogeneous_energy_density",
    "homogeneous_pressure",
    "hubble_derivative",
    "initial_hubble_from_friedmann",
    "simulate_flat_flrw",
]
