"""
Information-theoretic extensions for AgencityLab.

Includes:
    - Shannon entropy
    - Landauer principle
    - Agencity-information bridge
    - (optional) Vopson mass equivalence
"""

from .agencity_info import (
    agencity_information_index,
    agencity_information_density,
    agencity_structural_information,
    agencity_phase_information,
    full_information_summary,
)

from .landauer import (
    landauer_lower_bound,
    landauer_from_entropy,
)

from .shannon import (
    shannon_entropy,
    shannon_entropy_from_signal,
    conditional_entropy,
)

from .vopson import (
    information_mass,
    vopson_mass_equivalent,
)

__all__ = [
    # Agencity-specific
    "agencity_information_index",
    "agencity_information_density",
    "agencity_structural_information",
    "agencity_phase_information",
    "full_information_summary",

    # Shannon
    "shannon_entropy",
    "shannon_entropy_from_signal",
    "conditional_entropy",

    # Landauer
    "landauer_lower_bound",
    "landauer_from_entropy",

    # Vopson
    "information_mass",
    "vopson_mass_equivalent",
]