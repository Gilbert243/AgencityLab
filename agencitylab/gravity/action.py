"""Action-density evaluators for classical Agencity gravity.

The formulas implemented here follow Volume 2, Chapter 19, with the compact
Chapter-23 gauge notation supported only through an externally supplied U(1)
gauge potential ``A_mu``.  No gauge-field action or autonomous gauge dynamics
is implemented.

All functions use the Gravity convention ``(-,+,+,+)`` through the metric
objects supplied by the caller.  Scientific status: research.
"""

from __future__ import annotations

import numpy as np

from agencitylab.fields.physics import QuarticAgencityPotential
from agencitylab.scientific_status import ScientificStatus

from .geometry import (
    _finite_numeric_array,
    _finite_real_array,
    _metric_for_shape,
    _real_if_numerically_close,
    _real_scalar_or_field,
)

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH
MINIMAL_COUPLING = 0.0
CONFORMAL_COUPLING_4D = 1.0 / 6.0


def _finite_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_gravitational_constant(value) -> float:
    result = _finite_scalar(value, name="gravitational_constant")
    if result <= 0.0:
        raise ValueError("gravitational_constant must be strictly positive")
    return result


def minimal_coupling() -> float:
    """Return the explicitly named minimal coupling ``xi = 0``."""

    return MINIMAL_COUPLING


def conformal_coupling() -> float:
    """Return ``xi = 1/6`` for the massless 4D Klein-Gordon conformal case.

    This is a named source-theory value, not a universal default for Agencity
    gravity calculations.
    """

    return CONFORMAL_COUPLING_4D


def covariant_scalar_derivative(phi, partial_derivatives, gauge_field=None) -> np.ndarray:
    """Evaluate the scalar derivative, optionally with external U(1) ``A_mu``.

    For a scalar field without a gauge potential, ``nabla_mu phi = partial_mu
    phi``.  If an external gauge field is supplied, Chapter 23 defines
    ``D_mu phi = partial_mu phi - i A_mu phi``.

    ``partial_derivatives`` must have shape ``phi.shape + (4,)``.  A gauge
    field may have shape ``(4,)`` (constant over the field) or the exact same
    derivative shape.  No ambiguous general broadcasting is accepted.
    """

    field = _finite_numeric_array(phi, name="phi")
    derivatives = _finite_numeric_array(partial_derivatives, name="partial_derivatives")
    expected = field.shape + (4,)
    if derivatives.shape != expected:
        raise ValueError(
            f"partial_derivatives must have shape {expected} for phi shape {field.shape}"
        )
    if gauge_field is None:
        return np.array(derivatives, copy=True)

    gauge = _finite_real_array(gauge_field, name="gauge_field")
    if gauge.shape == (4,):
        gauge = np.broadcast_to(gauge, expected)
    elif gauge.shape != expected:
        raise ValueError(
            f"gauge_field must have shape (4,) or {expected}; got {gauge.shape}"
        )
    return derivatives - 1j * gauge * field[..., np.newaxis]


def matter_lagrangian_density(
    phi,
    derivatives,
    inverse_metric,
    potential: QuarticAgencityPotential,
    *,
    gauge_field=None,
):
    """Return the Chapter-19 matter Lagrangian ``L_phi`` before ``sqrt(-g)``.

    The evaluated source formula is
    ``1/2 g^{mu nu} D_mu(bar(phi)) D_nu(phi) - V(|phi|)``.
    ``derivatives`` are ordinary scalar derivatives; if ``gauge_field`` is
    supplied they are converted explicitly to the Chapter-23 ``D_mu`` form.
    """

    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    field = _finite_numeric_array(phi, name="phi")
    covariant = covariant_scalar_derivative(field, derivatives, gauge_field=gauge_field)
    metric = _metric_for_shape(inverse_metric, field.shape, name="inverse_metric")
    contraction = np.einsum(
        "...mn,...m,...n->...",
        metric,
        np.conjugate(covariant),
        covariant,
    )
    contraction = _real_if_numerically_close(
        contraction, name="g^{mu nu} D_mu(bar(phi)) D_nu(phi)"
    )
    result = 0.5 * contraction - potential.value(field)
    if not np.all(np.isfinite(result)):
        raise ValueError("matter Lagrangian density must be finite")
    return result


def matter_action_density(
    phi,
    derivatives,
    inverse_metric,
    sqrt_minus_g,
    potential: QuarticAgencityPotential,
    *,
    gauge_field=None,
):
    """Return the invariant matter action integrand ``sqrt(-g) * L_phi``."""

    field = _finite_numeric_array(phi, name="phi")
    measure = _real_scalar_or_field(sqrt_minus_g, field.shape, name="sqrt_minus_g")
    if np.any(measure < 0.0):
        raise ValueError("sqrt_minus_g must be non-negative")
    lagrangian = matter_lagrangian_density(
        field,
        derivatives,
        inverse_metric,
        potential,
        gauge_field=gauge_field,
    )
    return measure * lagrangian


def nonminimal_coupling_density(phi, sqrt_minus_g, scalar_curvature, *, xi):
    """Return the source nonminimal action integrand.

    Evaluates ``-1/2 * xi * sqrt(-g) * R * |phi|^2``.  ``xi`` is always
    explicit; no conformal value is silently selected.
    """

    field = _finite_numeric_array(phi, name="phi")
    xi_value = _finite_scalar(xi, name="xi")
    measure = _real_scalar_or_field(sqrt_minus_g, field.shape, name="sqrt_minus_g")
    curvature = _real_scalar_or_field(
        scalar_curvature, field.shape, name="scalar_curvature"
    )
    if np.any(measure < 0.0):
        raise ValueError("sqrt_minus_g must be non-negative")
    return -0.5 * xi_value * measure * curvature * np.abs(field) ** 2


def einstein_hilbert_density(sqrt_minus_g, scalar_curvature, *, gravitational_constant):
    """Return ``sqrt(-g) * R / (16*pi*G)``.

    Scalar inputs or identically shaped arrays are accepted.  General implicit
    broadcasting is rejected.
    """

    gravitational_constant_value = _positive_gravitational_constant(gravitational_constant)
    measure = _finite_real_array(sqrt_minus_g, name="sqrt_minus_g")
    curvature = _finite_real_array(scalar_curvature, name="scalar_curvature")
    if np.any(measure < 0.0):
        raise ValueError("sqrt_minus_g must be non-negative")
    if measure.shape == ():
        measure = np.broadcast_to(measure, curvature.shape)
    elif curvature.shape == ():
        curvature = np.broadcast_to(curvature, measure.shape)
    elif measure.shape != curvature.shape:
        raise ValueError(
            "sqrt_minus_g and scalar_curvature must be scalars or have identical shapes"
        )
    return measure * curvature / (16.0 * np.pi * gravitational_constant_value)


def total_gravity_field_lagrangian_density(
    phi,
    derivatives,
    inverse_metric,
    sqrt_minus_g,
    scalar_curvature,
    potential: QuarticAgencityPotential,
    *,
    xi,
    gravitational_constant,
    gauge_field=None,
):
    """Return the full Chapter-19/23 classical action integrand.

    This is the sum of Einstein-Hilbert, matter, and nonminimal-coupling
    integrands.  It is an evaluator only; no metric or field equation is solved.
    """

    field = _finite_numeric_array(phi, name="phi")
    measure = _real_scalar_or_field(sqrt_minus_g, field.shape, name="sqrt_minus_g")
    curvature = _real_scalar_or_field(
        scalar_curvature, field.shape, name="scalar_curvature"
    )
    matter = matter_action_density(
        field,
        derivatives,
        inverse_metric,
        measure,
        potential,
        gauge_field=gauge_field,
    )
    nonminimal = nonminimal_coupling_density(
        field,
        measure,
        curvature,
        xi=xi,
    )
    gravity = einstein_hilbert_density(
        measure,
        curvature,
        gravitational_constant=gravitational_constant,
    )
    return gravity + matter + nonminimal
