"""Landauer-related information-physics helpers."""

from __future__ import annotations

import numpy as np

from agencitylab.constants.physics import BOLTZMANN_CONSTANT

LN2 = float(np.log(2.0))


def _nonnegative_finite_scalar(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def landauer_lower_bound(bits: float, temperature: float) -> float:
    """Return ``k_B * T * ln(2) * bits`` in joules.

    ``BOLTZMANN_CONSTANT`` is a metadata-rich :class:`PhysicalConstant`; its
    numeric ``value`` is used explicitly.  Zero bits or zero temperature return
    zero, while negative or non-finite inputs are rejected.
    """

    bit_count = _nonnegative_finite_scalar(bits, name="bits")
    temperature_value = _nonnegative_finite_scalar(
        temperature,
        name="temperature",
    )
    return float(
        BOLTZMANN_CONSTANT.value * temperature_value * LN2 * bit_count
    )


def landauer_from_entropy(entropy_nats: float, temperature: float) -> float:
    """Convert non-negative entropy in nats to the Landauer energy bound."""

    entropy = _nonnegative_finite_scalar(entropy_nats, name="entropy_nats")
    return landauer_lower_bound(entropy / LN2, temperature)
