"""Agencial entropy definitions from the accepted theory documents.

Two different source quantities coexist and must not be conflated:

- Volume 2 Eq. (18.5): field agencial entropy of the autonomous field ``phi``;
- Volume 1 Appendix H Eq. (H.26): contrast agencial entropy of canonical ``J``.

Scientific status of this thermodynamic implementation: research.
"""

from __future__ import annotations

import warnings

import numpy as np

from agencitylab.fields.numerics import UniformRectilinearGrid, integrate_spatial
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_real_scalar(value, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return scalar


def _finite_field(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain real or complex numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def field_agencial_entropy(
    phi,
    a: float,
    grid: UniformRectilinearGrid,
) -> float:
    """Return ``(a / 2) * integral |phi|**2 dV`` from Volume 2 Eq. (18.5).

    The function supports real and complex fields and reuses the existing
    spatial quadrature contract.  No sign constraint is silently imposed on
    ``a``; the source statement ``S_ag >= 0`` follows under an appropriate
    positive-``a`` physical context.
    """

    field = _finite_field(phi, name="phi")
    coefficient = _finite_real_scalar(a, name="a")
    intensity = np.asarray(np.abs(field) ** 2, dtype=float)
    integral = integrate_spatial(intensity, grid)
    return float(0.5 * coefficient * integral)


def contrast_agencial_entropy(J, j_max: float, k_b: float):
    """Return Volume 1 Appendix H Eq. (H.26) exactly.

    The source definition is
    ``S_ag = -k_B * ln(1 - |J| / J_max)`` where ``J_max`` is a positive
    maximum-contrast scale, for example ``ln(1 + D_max/e)`` in the manuscript.
    The logarithm requires ``|J| < J_max`` for a finite real-valued result.
    This definition is distinct from :func:`field_agencial_entropy`.
    """

    contrast = np.asarray(J)
    if not np.issubdtype(contrast.dtype, np.number) or np.issubdtype(
        contrast.dtype, np.bool_
    ):
        raise TypeError("J must contain real numeric values")
    if np.iscomplexobj(contrast):
        raise ValueError("J must be real")
    contrast = np.asarray(contrast, dtype=float)
    if not np.all(np.isfinite(contrast)):
        raise ValueError("J must contain only finite values")

    maximum = _finite_real_scalar(j_max, name="j_max", positive=True)
    boltzmann = _finite_real_scalar(k_b, name="k_b", positive=True)
    ratio = np.abs(contrast) / maximum
    if np.any(ratio >= 1.0):
        raise ValueError("contrast entropy requires |J| < j_max")
    result = -boltzmann * np.log1p(-ratio)
    return float(result) if result.ndim == 0 else result


def agential_entropy(x):
    """Legacy Shannon-style placeholder unrelated to either accepted entropy.

    Kept only for backwards compatibility.  It is not an alias for either
    field or contrast agencial entropy because those definitions are
    physically and mathematically different.
    """

    warnings.warn(
        "agential_entropy is a deprecated Shannon-style placeholder; use "
        "field_agencial_entropy or contrast_agencial_entropy explicitly",
        DeprecationWarning,
        stacklevel=2,
    )
    values = np.asarray(x, dtype=float)
    probabilities = np.abs(values)
    probabilities = probabilities / (probabilities.sum() + 1e-12)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))
