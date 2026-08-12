"""Conditional Landauer relations explicitly stated in Volume 1.

These relations are not universal Agencity laws.  Volume 1 Appendix H presents
an exact equality only under the explicit construction ``P_c = k_B T_eff/tau``.
They are kept separate from the Chapter-18 field thermodynamics and retain
scientific status ``research``.
"""

from __future__ import annotations

import warnings

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _positive_scalar(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite strictly positive scalar")
    try:
        scalar = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite strictly positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return scalar


def landauer_characteristic_power(k_b: float, t_eff: float, tau: float) -> float:
    """Return ``P_c = k_B * T_eff / tau`` from Volume 1 Eq. (H.23)."""

    boltzmann = _positive_scalar(k_b, name="k_b")
    temperature = _positive_scalar(t_eff, name="t_eff")
    timescale = _positive_scalar(tau, name="tau")
    return boltzmann * temperature / timescale


def structural_information_rate(beta, tau: float):
    """Return ``I_dot_struct = |beta| / tau`` from Volume 1 Eq. (H.24)."""

    state = np.asarray(beta)
    if not np.issubdtype(state.dtype, np.number) or np.issubdtype(
        state.dtype, np.bool_
    ):
        raise TypeError("beta must contain real or complex numeric values")
    if not np.all(np.isfinite(state)):
        raise ValueError("beta must contain only finite values")
    timescale = _positive_scalar(tau, name="tau")
    result = np.asarray(np.abs(state) / timescale, dtype=float)
    return float(result) if result.ndim == 0 else result


def landauer_agencity_power(k_b: float, t_eff: float, information_rate):
    """Return ``|b| = k_B * T_eff * I_dot_struct`` from Volume 1 Eq. (H.25).

    The equality is conditional on the Volume-1 Landauer construction; this
    function does not redefine canonical ``b = P_c * beta``.
    """

    boltzmann = _positive_scalar(k_b, name="k_b")
    temperature = _positive_scalar(t_eff, name="t_eff")
    rate = np.asarray(information_rate)
    if not np.issubdtype(rate.dtype, np.number) or np.issubdtype(rate.dtype, np.bool_):
        raise TypeError("information_rate must contain real numeric values")
    if np.iscomplexobj(rate):
        raise ValueError("information_rate must be real")
    rate = np.asarray(rate, dtype=float)
    if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("information_rate must be finite and non-negative")
    result = boltzmann * temperature * rate
    return float(result) if result.ndim == 0 else result


def landauer_bound(kb, teff, istr):
    """Deprecated ambiguous legacy name for the conditional Volume-1 product."""

    warnings.warn(
        "landauer_bound is deprecated because the accepted source relation is "
        "a conditional equality; use landauer_agencity_power explicitly",
        DeprecationWarning,
        stacklevel=2,
    )
    return landauer_agencity_power(kb, teff, istr)
