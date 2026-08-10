"""
Landauer-related helpers for AgencityLab.
"""

from __future__ import annotations
import numpy as np

from agencitylab.constants.physics import BOLTZMANN_CONSTANT

LN2 = np.log(2.0)


def landauer_lower_bound(bits: float, temperature: float) -> float:
    """
    Landauer principle:

        E >= k_B * T * ln(2) * bits
    """
    bits = float(bits)
    temperature = float(temperature)

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    return BOLTZMANN_CONSTANT * temperature * LN2 * bits


def landauer_from_entropy(entropy_nats: float, temperature: float) -> float:
    """
    Convert entropy (nats) → energy using Landauer.

    1 nat = 1 / ln(2) bits
    """
    bits = entropy_nats / LN2
    return landauer_lower_bound(bits, temperature)