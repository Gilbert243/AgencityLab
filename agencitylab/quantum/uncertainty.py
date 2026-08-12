"""Uncertainty relation for the speculative quantised Agencity field.

Volume 2, Chapter 21 derives the bound from the canonically normalised field
under the constant-parameter bridge ``b = sqrt(P_c/tau) * phi``. This helper
therefore accepts scalar ``P_c`` and ``tau`` only and does not infer a bound for
time-dependent bridge parameters, where differentiating ``b`` would introduce
additional terms.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


def _finite_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def agencity_uncertainty_lower_bound(*, characteristic_power, tau, hbar) -> float:
    """Return the Chapter-21 lower bound ``(hbar/2) * P_c / tau``.

    ``P_c = 0`` is valid and yields an exact zero bound. ``tau`` and ``hbar``
    must be strictly positive. No value of ``hbar`` is silently assumed.
    """
    power = _finite_scalar(characteristic_power, name="characteristic_power")
    structural_time = _finite_scalar(tau, name="tau")
    planck = _finite_scalar(hbar, name="hbar")
    if power < 0.0:
        raise ValueError("characteristic_power must be non-negative")
    if structural_time <= 0.0:
        raise ValueError("tau must be strictly positive")
    if planck <= 0.0:
        raise ValueError("hbar must be strictly positive")
    return 0.5 * planck * power / structural_time
