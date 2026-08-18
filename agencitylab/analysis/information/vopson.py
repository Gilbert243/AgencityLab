"""Vopson information-mass equivalence hypothesis.

This module implements the proposed relation as a speculative information-physics
extension. It is not part of the canonical Theory of Agencity and is not treated
as an experimentally established mass law by AgencityLab.
"""

from __future__ import annotations

from agencitylab.constants.physics import SPEED_OF_LIGHT
from agencitylab.scientific_status import ScientificStatus

from .landauer import landauer_from_entropy, landauer_lower_bound

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE
C = SPEED_OF_LIGHT.value


def information_mass(bits: float, temperature: float) -> float:
    """Return the Vopson-hypothesis mass equivalent for ``bits`` at ``temperature``.

    The implemented relation is ``m = k_B*T*ln(2)*bits/c**2``. The Landauer
    helper supplies the numerical Boltzmann constant and validates finite,
    non-negative inputs; division by ``c**2`` is the speculative mass-equivalence
    step.
    """

    return float(landauer_lower_bound(bits, temperature) / (C**2))


def vopson_mass_equivalent(entropy_nats: float, temperature: float) -> float:
    """Return the speculative mass equivalent for entropy expressed in nats."""

    return float(landauer_from_entropy(entropy_nats, temperature) / (C**2))
