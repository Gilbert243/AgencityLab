"""Classical autonomous Agencity field dynamics.

This research subpackage connects the existing field Physics, Numerics, and
model contracts.  It is intentionally not re-exported from
``agencitylab.fields`` or the top-level package in this v1.1.x branch.
"""

from agencitylab.scientific_status import ScientificStatus

from .dissipative import dissipative_klein_gordon_acceleration
from .klein_gordon import klein_gordon_acceleration
from .simulation import (
    simulate_dissipative_klein_gordon,
    simulate_klein_gordon,
    simulate_tdgl,
)
from .tdgl import tdgl_rhs

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "SCIENTIFIC_STATUS",
    "dissipative_klein_gordon_acceleration",
    "klein_gordon_acceleration",
    "simulate_dissipative_klein_gordon",
    "simulate_klein_gordon",
    "simulate_tdgl",
    "tdgl_rhs",
]
