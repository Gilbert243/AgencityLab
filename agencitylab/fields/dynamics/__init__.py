"""Classical autonomous Agencity field dynamics.

This research subpackage connects the existing field Physics, Numerics, and
model contracts. Its public names are re-exported from ``agencitylab.fields``
for namespace-level convenience, while the package root remains intentionally small.
"""

from agencitylab.scientific_status import ScientificStatus

from .dissipative import dissipative_klein_gordon_acceleration
from .klein_gordon import klein_gordon_acceleration
from .simulation import (
    FLAT_FIELD_METRIC_SIGNATURE,
    simulate_dissipative_klein_gordon,
    simulate_klein_gordon,
    simulate_tdgl,
)
from .tdgl import tdgl_rhs

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH

__all__ = [
    "SCIENTIFIC_STATUS",
    "FLAT_FIELD_METRIC_SIGNATURE",
    "dissipative_klein_gordon_acceleration",
    "klein_gordon_acceleration",
    "simulate_dissipative_klein_gordon",
    "simulate_klein_gordon",
    "simulate_tdgl",
    "tdgl_rhs",
]
