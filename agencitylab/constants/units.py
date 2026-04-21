"""
Unit helpers for the AgencityLab framework.

The canonical information unit is the nat. The Bemwiz is defined as the
Agencity-specific output unit for b(t) in the reference theory.
"""

from __future__ import annotations

from math import log
from typing import Final

NAT_UNIT_SYMBOL: Final[str] = "nat"
AGENCITY_UNIT_SYMBOL: Final[str] = "Bz"

BIT_TO_NAT: Final[float] = log(2.0)
NAT_TO_BIT: Final[float] = 1.0 / BIT_TO_NAT
BEMWIZ_TO_NAT: Final[float] = 1.0


def nat_to_bemwiz(value: float) -> float:
    """Convert a value expressed in nats to Bemwiz units."""
    return float(value) / BEMWIZ_TO_NAT


def bemwiz_to_nat(value: float) -> float:
    """Convert a value expressed in Bemwiz units to nats."""
    return float(value) * BEMWIZ_TO_NAT
