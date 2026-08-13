"""Unified Agencity thermodynamics research API.

This package-local API has scientific status ``research``. The 1.0 software
contract exposes accepted thermodynamic research functions through the explicit
``agencitylab.thermodynamics`` namespace; pre-1.0 placeholder helpers are not
part of the package.
"""

from agencitylab.scientific_status import ScientificStatus

from .dissipation import (
    dissipation_density,
    entropy_production_density,
    total_dissipated_power,
    total_entropy_production,
)
from .effective_temperature import temperature_dependent_lambda
from .energy_balance import energy_balance_residual
from .entropy import contrast_agencial_entropy, field_agencial_entropy
from .landauer import (
    landauer_agencity_power,
    landauer_characteristic_power,
    structural_information_rate,
)
from .laws import (
    PhaseLawFit,
    modulus_law_margin,
    modulus_law_satisfied,
    phase_law_prediction,
    phase_law_residual,
    phi_imaginary_component,
    second_law_residual,
    thermal_reference_phase_fit,
)

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "SCIENTIFIC_STATUS",
    "PhaseLawFit",
    "contrast_agencial_entropy",
    "dissipation_density",
    "energy_balance_residual",
    "entropy_production_density",
    "field_agencial_entropy",
    "landauer_agencity_power",
    "landauer_characteristic_power",
    "modulus_law_margin",
    "modulus_law_satisfied",
    "phase_law_prediction",
    "phase_law_residual",
    "phi_imaginary_component",
    "second_law_residual",
    "structural_information_rate",
    "temperature_dependent_lambda",
    "thermal_reference_phase_fit",
    "total_dissipated_power",
    "total_entropy_production",
]
