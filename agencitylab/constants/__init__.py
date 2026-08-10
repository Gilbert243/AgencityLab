"""
Physical constants, reference values, characteristic system parameters,
A_ref registry, activity-factor registry and unit helpers
used by AgencityLab.
"""

# ============================================================
# PHYSICS
# ============================================================

from .physics import (
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    REDUCED_PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT,
    ELEMENTARY_CHARGE,
    AVOGADRO_CONSTANT,
    GAS_CONSTANT,
    GRAVITATIONAL_CONSTANT,
    PLANCK_TIME,
    PLANCK_LENGTH,
    LANDAUER_ENERGY,
)

# ============================================================
# REFERENCE VALUES
# ============================================================

from .reference_values import (
    DEFAULT_EPSILON,
    DEFAULT_TAU_THRESHOLD,
    DEFAULT_ACTIVITY_WINDOW,
    DEFAULT_CRM_WINDOW,
    DEFAULT_INFORMATION_SCALE,
    DEFAULT_POWER_SCALE,
    DEFAULT_BEMWIZ_SCALE,
)

# ============================================================
# A_ref (CANONICAL NORMALIZATION)
# ============================================================

from .reference_amplitudes import (
    CANONICAL_AREF_BY_DOMAIN_KIND,
    CANONICAL_AREF_BY_KIND,
    CANONICAL_AREF_BY_UNIT,
    register_reference_amplitude,
    reference_context_from_metadata,
    resolve_reference_amplitude,
    resolve_reference_amplitudes,
)

# ============================================================
# ACTIVITY FACTORS
# ============================================================

from .activity_factors import (
    CANONICAL_ACTIVITY_BY_MECHANISM,
    CANONICAL_ACTIVITY_BY_DOMAIN,
    CANONICAL_ACTIVITY_BY_DOMAIN_MECHANISM,
    register_activity_factor,
    resolve_activity_factor,
    activity_context_from_metadata,
)

# ============================================================
# CHARACTERISTIC TIMES
# ============================================================

from .characteristic_times import (
    CANONICAL_TAU_BY_SYSTEM,
    CANONICAL_TAU_BY_DOMAIN,
    register_characteristic_time,
    resolve_characteristic_time,
    tau_context_from_metadata,
)

# ============================================================
# CHARACTERISTIC POWERS
# ============================================================

from .characteristic_powers import (
    CANONICAL_POWER_BY_SYSTEM,
    CANONICAL_POWER_BY_DOMAIN,
    register_characteristic_power,
    resolve_characteristic_power,
    power_context_from_metadata,
)

# ============================================================
# UNITS
# ============================================================

from .units import (
    NAT,
    BIT,
    JOULE,
    WATT,
    BEMWIZ,
    bit_to_nat,
    nat_to_bit,
    bemwiz_to_nat,
    nat_to_bemwiz,
    bemwiz_to_watt,
    watt_to_bemwiz,
    convert,
    AGENCITY_UNIT_SYMBOL,
    NAT_UNIT_SYMBOL,
)

# ============================================================
# PUBLIC API
# ============================================================

__all__ = [

    # ========================================================
    # physics
    # ========================================================

    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "REDUCED_PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "ELEMENTARY_CHARGE",
    "AVOGADRO_CONSTANT",
    "GAS_CONSTANT",
    "GRAVITATIONAL_CONSTANT",
    "PLANCK_TIME",
    "PLANCK_LENGTH",
    "LANDAUER_ENERGY",

    # ========================================================
    # reference values
    # ========================================================

    "DEFAULT_EPSILON",
    "DEFAULT_TAU_THRESHOLD",
    "DEFAULT_ACTIVITY_WINDOW",
    "DEFAULT_CRM_WINDOW",
    "DEFAULT_INFORMATION_SCALE",
    "DEFAULT_POWER_SCALE",
    "DEFAULT_BEMWIZ_SCALE",

    # ========================================================
    # A_ref
    # ========================================================

    "CANONICAL_AREF_BY_DOMAIN_KIND",
    "CANONICAL_AREF_BY_KIND",
    "CANONICAL_AREF_BY_UNIT",

    "register_reference_amplitude",

    "reference_context_from_metadata",

    "resolve_reference_amplitude",
    "resolve_reference_amplitudes",

    # ========================================================
    # activity factors
    # ========================================================

    "CANONICAL_ACTIVITY_BY_MECHANISM",
    "CANONICAL_ACTIVITY_BY_DOMAIN",
    "CANONICAL_ACTIVITY_BY_DOMAIN_MECHANISM",

    "register_activity_factor",

    "resolve_activity_factor",

    "activity_context_from_metadata",

    # ========================================================
    # characteristic times
    # ========================================================

    "CANONICAL_TAU_BY_SYSTEM",
    "CANONICAL_TAU_BY_DOMAIN",

    "register_characteristic_time",

    "resolve_characteristic_time",

    "tau_context_from_metadata",

    # ========================================================
    # characteristic powers
    # ========================================================

    "CANONICAL_POWER_BY_SYSTEM",
    "CANONICAL_POWER_BY_DOMAIN",

    "register_characteristic_power",

    "resolve_characteristic_power",

    "power_context_from_metadata",

    # ========================================================
    # units
    # ========================================================

    "NAT",
    "BIT",
    "JOULE",
    "WATT",
    "BEMWIZ",

    "bit_to_nat",
    "nat_to_bit",

    "bemwiz_to_nat",
    "nat_to_bemwiz",

    "bemwiz_to_watt",
    "watt_to_bemwiz",

    "convert",

    "AGENCITY_UNIT_SYMBOL",
    "NAT_UNIT_SYMBOL",
]