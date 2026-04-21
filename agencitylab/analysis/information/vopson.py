"""
Vopson-inspired helpers for AgencityLab.

This module provides lightweight information-mass estimators based on
the idea that information may be assigned a physically meaningful mass
equivalent in specific theoretical frameworks.
"""

from __future__ import annotations

import numpy as np


def information_mass(bits: float, alpha: float = 1.0) -> float:
    """
    Return a simple information-mass proxy.

    The parameter alpha allows the user to scale the estimate.
    """
    return float(alpha) * float(bits)


def vopson_mass_equivalent(bits: float, mass_per_bit: float = 1e-38) -> float:
    """
    Return a linear information-mass estimate.

    This is a placeholder helper for research workflows; the coefficient can
    be adjusted by the user.
    """
    return float(bits) * float(mass_per_bit)
