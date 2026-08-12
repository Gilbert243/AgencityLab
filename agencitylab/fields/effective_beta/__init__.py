"""Research implementation of the Volume-2 Chapter-15 effective beta field.

This package is intentionally separate from both observable ``beta_obs`` and
the autonomous Chapter-16 ``phi`` field.
"""

from agencitylab.scientific_status import ScientificStatus

from .equation import effective_beta_rhs
from .reaction import effective_beta_reaction, effective_beta_stationary_amplitude

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "SCIENTIFIC_STATUS",
    "effective_beta_reaction",
    "effective_beta_rhs",
    "effective_beta_stationary_amplitude",
]
