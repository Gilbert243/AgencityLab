"""Speculative quantum-field primitives for Agencity.

This package implements the limited Chapter-21 contracts that are directly
operational without inventing a full quantum-field-theory framework. It does
not quantise the canonical observable pipeline and makes no experimental claim
for agentons or quantum Agencity.
"""

from agencitylab.scientific_status import ScientificStatus

from .fock import (
    annihilation_operator,
    creation_operator,
    fock_state,
    number_operator,
    occupation_expectation,
    truncated_commutator,
    truncation_commutator_defect,
    vacuum_state,
)
from .modes import (
    broken_symmetry_vacuum_amplitude,
    goldstone_angular_frequency,
    goldstone_mass_squared,
    radial_angular_frequency,
    radial_mass,
    radial_mass_squared,
)
from .propagators import goldstone_propagator, radial_propagator
from .renormalization import one_loop_quartic_beta
from .uncertainty import agencity_uncertainty_lower_bound

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE

__all__ = [
    "SCIENTIFIC_STATUS",
    "agencity_uncertainty_lower_bound",
    "annihilation_operator",
    "broken_symmetry_vacuum_amplitude",
    "creation_operator",
    "fock_state",
    "goldstone_angular_frequency",
    "goldstone_mass_squared",
    "goldstone_propagator",
    "number_operator",
    "occupation_expectation",
    "one_loop_quartic_beta",
    "radial_angular_frequency",
    "radial_mass",
    "radial_mass_squared",
    "radial_propagator",
    "truncated_commutator",
    "truncation_commutator_defect",
    "vacuum_state",
]
