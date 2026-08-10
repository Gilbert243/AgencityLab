"""
Advanced unit system for AgencityLab.

Canonical relation:
    1 Bemwiz (Bz) = 1 Watt × 1 nat
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, Final

# Optional safeguard (fallback if not available)
try:
    from agencitylab.core.safeguards import EPS
except Exception:
    EPS = 1e-12


# ============================================================
# STRUCTURE
# ============================================================

@dataclass(frozen=True)
class Unit:
    name: str
    symbol: str
    dimension: str
    description: str


# ============================================================
# BASE UNITS
# ============================================================

NAT = Unit("nat", "nat", "information", "Natural information unit")
BIT = Unit("bit", "bit", "information", "Binary information unit")

JOULE = Unit("joule", "J", "energy", "Energy")
WATT = Unit("watt", "W", "power", "Power")

BEMWIZ = Unit("Bemwiz", "Bz", "power × information", "Agencity unit (W·nat)")


# ============================================================
# FUNDAMENTAL CONVERSIONS
# ============================================================

BIT_TO_NAT: Final[float] = log(2.0)
NAT_TO_BIT: Final[float] = 1.0 / BIT_TO_NAT

# Canonical symbolic identity (no scaling)
BEMWIZ_TO_NAT: Final[float] = 1.0
NAT_TO_BEMWIZ: Final[float] = 1.0
BEMWIZ_TO_WATT: Final[float] = 1.0
WATT_TO_BEMWIZ: Final[float] = 1.0


# ============================================================
# BASIC CONVERSIONS
# ============================================================

def bit_to_nat(x: float) -> float:
    return float(x) * BIT_TO_NAT


def nat_to_bit(x: float) -> float:
    return float(x) * NAT_TO_BIT


# ============================================================
# BEMWIZ CONVERSIONS
# ============================================================

def bemwiz_to_nat(value: float, power: float = 1.0) -> float:
    """
    Convert Bemwiz to nat under a given power convention.

    Since:
        Bz = W × nat

    nat = Bz / W
    """
    power = max(float(power), EPS)
    return float(value) / power


def nat_to_bemwiz(value: float, power: float = 1.0) -> float:
    power = max(float(power), EPS)
    return float(value) * power


def bemwiz_to_watt(value: float, nat_value: float = 1.0) -> float:
    nat_value = max(float(nat_value), EPS)
    return float(value) / nat_value


def watt_to_bemwiz(power: float, nat_value: float = 1.0) -> float:
    return float(power) * float(nat_value)


# ============================================================
# GENERIC CONVERSION
# ============================================================

def convert(value, from_unit: Unit, to_unit: Unit):
    f = from_unit.symbol.lower()
    t = to_unit.symbol.lower()

    if f == "bit" and t == "nat":
        return bit_to_nat(value)
    if f == "nat" and t == "bit":
        return nat_to_bit(value)

    if f == "bz" and t == "w":
        return bemwiz_to_watt(value)
    if f == "w" and t == "bz":
        return watt_to_bemwiz(value)

    if f == "bz" and t == "nat":
        return bemwiz_to_nat(value)
    if f == "nat" and t == "bz":
        return nat_to_bemwiz(value)

    raise ValueError(f"Unsupported conversion: {f} → {t}")


# ============================================================
# PREFIXES
# ============================================================

PREFIXES: Dict[str, float] = {
    "y": 1e-24,
    "z": 1e-21,
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1.0,
    "da": 1e1,
    "h": 1e2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "Z": 1e21,
    "Y": 1e24,
}


def to_prefixed_bemwiz(value: float, prefix: str):
    return float(value) / PREFIXES[prefix]


def from_prefixed_bemwiz(value: float, prefix: str):
    return float(value) * PREFIXES[prefix]


# ============================================================
# UTILS
# ============================================================

def describe():
    return {
        "nat": NAT.description,
        "bemwiz": BEMWIZ.description,
    }


# ============================================================
# EXPORTS
# ============================================================

AGENCITY_UNIT_SYMBOL = BEMWIZ.symbol
AGENCY_UNIT_SYMBOL = AGENCITY_UNIT_SYMBOL  # backward compatibility
NAT_UNIT_SYMBOL = NAT.symbol


__all__ = [
    "Unit",
    "NAT",
    "BIT",
    "JOULE",
    "WATT",
    "BEMWIZ",

    "BIT_TO_NAT",
    "NAT_TO_BIT",
    "BEMWIZ_TO_NAT",
    "NAT_TO_BEMWIZ",
    "BEMWIZ_TO_WATT",
    "WATT_TO_BEMWIZ",

    "bit_to_nat",
    "nat_to_bit",
    "bemwiz_to_nat",
    "nat_to_bemwiz",
    "bemwiz_to_watt",
    "watt_to_bemwiz",

    "convert",

    "PREFIXES",
    "to_prefixed_bemwiz",
    "from_prefixed_bemwiz",

    "AGENCITY_UNIT_SYMBOL",
    "AGENCY_UNIT_SYMBOL",
    "NAT_UNIT_SYMBOL",
]