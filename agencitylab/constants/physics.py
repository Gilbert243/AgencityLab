"""
Selected physical constants.

The values are taken from CODATA 2022-recommended constants.
They are provided here as stable defaults for scientific computations.
"""

from __future__ import annotations

SPEED_OF_LIGHT = 299_792_458.0  # m / s
PLANCK_CONSTANT = 6.626_070_15e-34  # J s
REDUCED_PLANCK_CONSTANT = PLANCK_CONSTANT / (2.0 * 3.141592653589793)  # J s
BOLTZMANN_CONSTANT = 1.380_649e-23  # J / K
ELEMENTARY_CHARGE = 1.602_176_634e-19  # C
AVOGADRO_CONSTANT = 6.022_140_76e23  # 1 / mol
GAS_CONSTANT = AVOGADRO_CONSTANT * BOLTZMANN_CONSTANT  # J / (mol K)
GRAVITATIONAL_CONSTANT = 6.674_30e-11  # m^3 / (kg s^2)
