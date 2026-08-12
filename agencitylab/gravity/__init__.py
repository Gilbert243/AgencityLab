"""Classical Agencity gravity research primitives.

This package implements the limited numerical contract stated in Volume 2,
Chapter 19, with the compact Chapter-23 external-gauge notation where useful.
The Gravity convention is explicitly ``(-,+,+,+)`` and is not silently
identified with the Chapter-16 ``(+,-,-,-)`` flat-field convention.

No Einstein solver, cosmological solver, symbolic GR engine, gauge dynamics,
quantum field machinery, or thermodynamics is provided here.
"""

from agencitylab.scientific_status import ScientificStatus

from .action import (
    CONFORMAL_COUPLING_4D,
    MINIMAL_COUPLING,
    conformal_coupling,
    covariant_scalar_derivative,
    einstein_hilbert_density,
    matter_action_density,
    matter_lagrangian_density,
    minimal_coupling,
    nonminimal_coupling_density,
    total_gravity_field_lagrangian_density,
)
from .equations import curved_field_residual, einstein_equation_residual, minkowski_box
from .geometry import (
    GRAVITY_METRIC_SIGNATURE,
    metric_with_perturbation,
    minkowski_inverse_metric,
    minkowski_metric,
    sqrt_minus_g,
)
from .stress_energy import stress_energy_tensor

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "SCIENTIFIC_STATUS",
    "GRAVITY_METRIC_SIGNATURE",
    "MINIMAL_COUPLING",
    "CONFORMAL_COUPLING_4D",
    "conformal_coupling",
    "covariant_scalar_derivative",
    "curved_field_residual",
    "einstein_equation_residual",
    "einstein_hilbert_density",
    "matter_action_density",
    "matter_lagrangian_density",
    "metric_with_perturbation",
    "minimal_coupling",
    "minkowski_box",
    "minkowski_inverse_metric",
    "minkowski_metric",
    "nonminimal_coupling_density",
    "sqrt_minus_g",
    "stress_energy_tensor",
    "total_gravity_field_lagrangian_density",
]
