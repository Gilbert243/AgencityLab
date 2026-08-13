"""Chapter-15 effective reaction term for the Agencity state field ``beta``.

This module implements the postulated Ginzburg--Landau reaction term from
Volume 2, Eq. (15.3),

    R(beta) = a beta - b |beta|^2 beta,

without identifying the source coefficient ``b`` with the canonical Agencity
flux ``b(t)``.  The public API therefore uses the unambiguous names
``linear_coefficient`` and ``saturation_coefficient``.

Scientific status: research.  This effective beta-field model is distinct from
both the observable ``beta_obs`` computed from ``u`` and the autonomous
canonically normalised field ``phi`` used from Chapter 16 onward.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_real_scalar(value, *, name: str) -> float:
    try:
        scalar = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def effective_beta_reaction(
    beta,
    *,
    linear_coefficient: float,
    saturation_coefficient: float,
) -> np.ndarray:
    """Return the Volume-2 Eq. (15.3) reaction term.

    Parameters
    ----------
    beta:
        Real or complex effective beta field.
    linear_coefficient:
        Source coefficient ``a`` in Eq. (15.3).  It may have either sign.
    saturation_coefficient:
        Source coefficient denoted ``b`` in Eq. (15.3).  It must be strictly
        positive.  The explicit name avoids collision with the canonical
        Agencity flux ``b = P_c beta``.
    """

    field = np.asarray(beta)
    if field.size == 0:
        raise ValueError("beta must not be empty")
    if not np.issubdtype(field.dtype, np.number) or np.issubdtype(field.dtype, np.bool_):
        raise TypeError("beta must contain real or complex numeric values")
    if not np.all(np.isfinite(field)):
        raise ValueError("beta must contain only finite values")

    linear = _finite_real_scalar(linear_coefficient, name="linear_coefficient")
    saturation = _finite_real_scalar(
        saturation_coefficient,
        name="saturation_coefficient",
    )
    if saturation <= 0.0:
        raise ValueError("saturation_coefficient must be strictly positive")

    return linear * field - saturation * np.abs(field) ** 2 * field


def effective_beta_stationary_amplitude(
    *,
    linear_coefficient: float,
    saturation_coefficient: float,
) -> float:
    """Return ``sqrt(a/b)`` for the broken stationary branch of Eq. (15.3).

    Volume 2 states that the non-zero homogeneous branch exists for ``a > 0``
    and ``b > 0``.  For ``a <= 0`` this helper refuses to manufacture a
    non-zero stationary amplitude.
    """

    linear = _finite_real_scalar(linear_coefficient, name="linear_coefficient")
    saturation = _finite_real_scalar(
        saturation_coefficient,
        name="saturation_coefficient",
    )
    if saturation <= 0.0:
        raise ValueError("saturation_coefficient must be strictly positive")
    if linear <= 0.0:
        raise ValueError(
            "the non-zero Chapter-15 stationary amplitude requires linear_coefficient > 0"
        )
    return float(np.sqrt(linear / saturation))
