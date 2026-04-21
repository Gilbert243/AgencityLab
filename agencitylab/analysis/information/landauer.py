"""
Landauer-related helpers for AgencityLab.

The functions here provide lower-bound estimates inspired by the Landauer
principle when a temperature is available.
"""

from __future__ import annotations

import numpy as np

from agencitylab.constants.physics import BOLTZMANN_CONSTANT


def landauer_lower_bound(bits: float, temperature: float) -> float:
    """
    Compute the Landauer lower bound for energy dissipation.

    E >= k_B * T * ln(2) * bits
    """
    bits = float(bits)
    temperature = float(temperature)
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    return BOLTZMANN_CONSTANT * temperature * np.log(2.0) * bits
