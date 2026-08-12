"""Chapter-16 flat-spacetime conservation-law primitives for ``phi``.

Volume 2 Chapter 16 uses Minkowski signature ``(+,-,-,-)`` and gives the
symmetric energy--momentum tensor (16.4) and global-U(1) Noether current
(16.5).  This module keeps that convention explicit and separate from the
Chapter-19 gravity package, which uses ``(-,+,+,+)``.

Scientific status: research.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from .potential import QuarticAgencityPotential

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH
FLAT_FIELD_METRIC_SIGNATURE = (1, -1, -1, -1)


def flat_field_minkowski_metric() -> np.ndarray:
    """Return the Chapter-16 ``(+,-,-,-)`` Minkowski metric."""

    return np.diag(np.asarray(FLAT_FIELD_METRIC_SIGNATURE, dtype=float))


def _finite_numeric_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain real or complex numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _field_and_derivatives(phi, derivatives) -> tuple[np.ndarray, np.ndarray]:
    field = _finite_numeric_array(phi, name="phi")
    deriv = _finite_numeric_array(derivatives, name="derivatives")
    expected = field.shape + (4,)
    if deriv.shape != expected:
        raise ValueError(f"derivatives must have shape {expected}")
    return field, deriv


def _raised_derivatives(derivatives: np.ndarray) -> np.ndarray:
    signature = np.asarray(FLAT_FIELD_METRIC_SIGNATURE, dtype=float)
    return derivatives * signature


def _real_if_close(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.iscomplexobj(array):
        return array
    scale = np.maximum(1.0, np.abs(np.real(array)))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(np.abs(np.imag(array)) > tolerance):
        raise ValueError(f"{name} has a significant imaginary component")
    return np.real(array)


def flat_field_lagrangian_density(
    phi,
    derivatives,
    potential: QuarticAgencityPotential,
) -> np.ndarray:
    """Return the Chapter-16 Lagrangian density, Eq. (16.1).

    ``derivatives[..., mu]`` are the lower-index coordinate derivatives
    ``partial_mu phi`` ordered as time followed by the three spatial
    coordinates.  Natural units ``c = 1`` are used, matching the existing
    autonomous-field layer.
    """

    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    field, deriv = _field_and_derivatives(phi, derivatives)
    raised = _raised_derivatives(deriv)
    contraction = np.sum(np.conjugate(deriv) * raised, axis=-1)
    lagrangian = 0.5 * contraction - potential.value(field)
    return _real_if_close(lagrangian, name="flat-field Lagrangian density")


def flat_energy_momentum_tensor(
    phi,
    derivatives,
    potential: QuarticAgencityPotential,
) -> np.ndarray:
    """Return the symmetric Chapter-16 energy--momentum tensor, Eq. (16.4).

    The returned last two axes are contravariant ``(mu, nu)`` components in
    the explicit ``(+,-,-,-)`` convention.
    """

    field, deriv = _field_and_derivatives(phi, derivatives)
    raised = _raised_derivatives(deriv)
    lagrangian = flat_field_lagrangian_density(field, deriv, potential)

    first = raised[..., :, np.newaxis] * np.conjugate(raised[..., np.newaxis, :])
    second = raised[..., np.newaxis, :] * np.conjugate(raised[..., :, np.newaxis])
    symmetric = 0.5 * (first + second)
    metric = flat_field_minkowski_metric()
    tensor = symmetric - metric * lagrangian[..., np.newaxis, np.newaxis]
    tensor = _real_if_close(tensor, name="flat-field energy-momentum tensor")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("energy-momentum tensor must be finite")
    return tensor


def u1_noether_current(phi, derivatives) -> np.ndarray:
    """Return the Chapter-16 global-U(1) Noether current, Eq. (16.5).

    Implements ``J^mu = Im(phi * partial^mu conjugate(phi))`` using the
    Chapter-16 ``(+,-,-,-)`` metric.  This is a field-theory current and is not
    the canonical logarithmic contrast ``J(t)``.
    """

    field, deriv = _field_and_derivatives(phi, derivatives)
    raised = _raised_derivatives(deriv)
    return np.imag(field[..., np.newaxis] * np.conjugate(raised))


def phase_noether_current(amplitude, theta_derivatives) -> np.ndarray:
    """Return ``R^2 partial^mu Theta`` from the Chapter-16 phase decomposition.

    This implements the current appearing in Eq. (16.7).  Conservation itself
    is an on-shell statement; this helper does not force a supplied numerical
    field to satisfy ``partial_mu J^mu = 0``.
    """

    radius = np.asarray(amplitude, dtype=float)
    if radius.size == 0 or not np.all(np.isfinite(radius)):
        raise ValueError("amplitude must be non-empty and finite")
    if np.any(radius < 0.0):
        raise ValueError("amplitude must be non-negative")
    theta_deriv = np.asarray(theta_derivatives, dtype=float)
    expected = radius.shape + (4,)
    if theta_deriv.shape != expected:
        raise ValueError(f"theta_derivatives must have shape {expected}")
    if not np.all(np.isfinite(theta_deriv)):
        raise ValueError("theta_derivatives must contain only finite values")
    return radius[..., np.newaxis] ** 2 * _raised_derivatives(theta_deriv)


def radial_equation_residual(
    amplitude,
    box_amplitude,
    phase_gradient_contraction,
    potential: QuarticAgencityPotential,
) -> np.ndarray:
    """Evaluate the Chapter-16 radial equation (16.6).

    Returns ``box(R) - R * (partial Theta)^2 + V'(R)``.  The radial derivative
    of the written quartic potential is exactly
    ``-lambda R + mu R^3``, which is the existing source-normalised
    ``potential.gradient(R)`` for real non-negative ``R``.
    """

    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    radius = np.asarray(amplitude, dtype=float)
    box = np.asarray(box_amplitude, dtype=float)
    contraction = np.asarray(phase_gradient_contraction, dtype=float)
    if radius.shape != box.shape or radius.shape != contraction.shape:
        raise ValueError("amplitude, box_amplitude, and phase_gradient_contraction must match")
    if radius.size == 0 or not np.all(np.isfinite(radius)):
        raise ValueError("amplitude must be non-empty and finite")
    if np.any(radius < 0.0):
        raise ValueError("amplitude must be non-negative")
    if not np.all(np.isfinite(box)) or not np.all(np.isfinite(contraction)):
        raise ValueError("radial-equation inputs must be finite")
    return box - radius * contraction + np.asarray(potential.gradient(radius), dtype=float)
