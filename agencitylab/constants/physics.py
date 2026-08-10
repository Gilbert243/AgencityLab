"""
Selected physical constants (CODATA-style + information physics).

Provides:
- SI constants
- Planck-scale constants
- Information-physics constants (Landauer)
- Structured metadata for scientific pipelines

All constants are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt, log


# ============================================================
# STRUCTURE
# ============================================================

@dataclass(frozen=True)
class PhysicalConstant:
    name: str
    symbol: str
    value: float
    unit: str
    description: str


# ============================================================
# FUNDAMENTAL CONSTANTS
# ============================================================

SPEED_OF_LIGHT = PhysicalConstant(
    "Speed of light", "c", 299_792_458.0, "m/s", "Speed of light in vacuum"
)

PLANCK_CONSTANT = PhysicalConstant(
    "Planck constant", "h", 6.626_070_15e-34, "J·s", "Quantum of action"
)

REDUCED_PLANCK_CONSTANT = PhysicalConstant(
    "Reduced Planck constant",
    "ħ",
    PLANCK_CONSTANT.value / (2.0 * pi),
    "J·s",
    "h / (2π)",
)

BOLTZMANN_CONSTANT = PhysicalConstant(
    "Boltzmann constant",
    "k_B",
    1.380_649e-23,
    "J/K",
    "Relates temperature to energy",
)

ELEMENTARY_CHARGE = PhysicalConstant(
    "Elementary charge",
    "e",
    1.602_176_634e-19,
    "C",
    "Charge of a proton",
)

AVOGADRO_CONSTANT = PhysicalConstant(
    "Avogadro constant",
    "N_A",
    6.022_140_76e23,
    "1/mol",
    "Particles per mole",
)

GAS_CONSTANT = PhysicalConstant(
    "Gas constant",
    "R",
    AVOGADRO_CONSTANT.value * BOLTZMANN_CONSTANT.value,
    "J/(mol·K)",
    "Universal gas constant",
)

GRAVITATIONAL_CONSTANT = PhysicalConstant(
    "Gravitational constant",
    "G",
    6.674_30e-11,
    "m^3/(kg·s^2)",
    "Newtonian gravity",
)


# ============================================================
# PLANCK SCALE
# ============================================================

PLANCK_LENGTH = PhysicalConstant(
    "Planck length",
    "l_P",
    sqrt((REDUCED_PLANCK_CONSTANT.value * GRAVITATIONAL_CONSTANT.value) / (SPEED_OF_LIGHT.value ** 3)),
    "m",
    "Fundamental length scale",
)

PLANCK_TIME = PhysicalConstant(
    "Planck time",
    "t_P",
    PLANCK_LENGTH.value / SPEED_OF_LIGHT.value,
    "s",
    "Fundamental time scale",
)


# ============================================================
# INFORMATION PHYSICS
# ============================================================

LANDAUER_ENERGY = PhysicalConstant(
    "Landauer energy per bit (1K)",
    "E_L",
    BOLTZMANN_CONSTANT.value * log(2.0),
    "J",
    "Minimum energy to erase one bit at 1K",
)


# ============================================================
# REGISTRY
# ============================================================

CONSTANTS = {
    "c": SPEED_OF_LIGHT,
    "h": PLANCK_CONSTANT,
    "hbar": REDUCED_PLANCK_CONSTANT,
    "kB": BOLTZMANN_CONSTANT,
    "e": ELEMENTARY_CHARGE,
    "NA": AVOGADRO_CONSTANT,
    "R": GAS_CONSTANT,
    "G": GRAVITATIONAL_CONSTANT,
    "lP": PLANCK_LENGTH,
    "tP": PLANCK_TIME,
    "EL": LANDAUER_ENERGY,
}


# ============================================================
# UTILS
# ============================================================

def get_constant(symbol: str) -> PhysicalConstant:
    if symbol not in CONSTANTS:
        raise KeyError(f"Unknown constant '{symbol}'")
    return CONSTANTS[symbol]


def list_constants():
    return list(CONSTANTS.keys())