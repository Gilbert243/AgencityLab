"""
Vopson information mass equivalence (advanced).
"""

from __future__ import annotations
import numpy as np

C = 299792458  # speed of light


def information_mass(bits: float, temperature: float) -> float:
    """
    Information → mass equivalence (Vopson hypothesis)

    m = (k_B * T * ln2 * bits) / c^2
    """
    from agencitylab.constants.physics import BOLTZMANN_CONSTANT

    return (BOLTZMANN_CONSTANT * temperature * np.log(2) * bits) / (C**2)


def vopson_mass_equivalent(entropy_nats: float, temperature: float):
    """
    Convert entropy (nats) → mass
    """
    bits = entropy_nats / np.log(2)
    return information_mass(bits, temperature)