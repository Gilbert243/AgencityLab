"""Equation residuals for the classical Chapter-19 gravity coupling.

This module evaluates equations only when the required geometric tensors are
supplied.  It does not compute Christoffel symbols, curvature tensors, solve the
Einstein equations, or evolve a metric.

Metric-convention warning
-------------------------
Chapter 19 explicitly adopts ``(-,+,+,+)`` whereas Chapter 16 and the existing
``fields.dynamics`` layer use ``(+,-,-,-)``.  The Chapter-19 curved field
equation is implemented literally.  No sign conversion is silently applied to
make it coincide with the Chapter-16 flat equation.

Scientific status: research.
"""

from __future__ import annotations

import numpy as np

from agencitylab.fields.physics import QuarticAgencityPotential
from agencitylab.scientific_status import ScientificStatus

from .action import _finite_scalar, _positive_gravitational_constant
from .geometry import (
    _finite_numeric_array,
    _metric_array,
    _real_scalar_or_field,
)

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def minkowski_box(phi_tt, spatial_laplacian):
    """Return the flat d'Alembertian for Gravity signature ``(-,+,+,+)``.

    In natural units this convention gives ``box(phi) = -phi_tt + nabla^2
    phi``.  This helper exists specifically to keep the Chapter-19 sign
    convention visible at call sites.
    """

    second_time = _finite_numeric_array(phi_tt, name="phi_tt")
    laplacian = _finite_numeric_array(spatial_laplacian, name="spatial_laplacian")
    if second_time.shape != laplacian.shape:
        raise ValueError("phi_tt and spatial_laplacian must have identical shapes")
    return -second_time + laplacian


def curved_field_residual(
    box_phi,
    phi,
    potential: QuarticAgencityPotential,
    scalar_curvature,
    *,
    xi,
):
    """Evaluate the Volume-2 Chapter-19 curved scalar-field residual.

    The source equation (19.6) is evaluated as

    ``box(phi) + potential.gradient(phi) - xi * R * phi``.

    Reusing ``potential.gradient`` avoids the formally singular
    ``V'(|phi|) * phi / |phi|`` representation at ``phi = 0`` without adding
    any epsilon.  This is a mathematical equivalence for the existing quartic
    potential, not a new physical regularisation.

    Note that the literal Chapter-19 formula combined with its ``(-,+,+,+)``
    signature does not become the Chapter-16 ``(+,-,-,-)`` field equation by a
    silent metric-sign swap.  That source-level convention difference is
    intentionally preserved and tested.
    """

    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    field = _finite_numeric_array(phi, name="phi")
    box = _finite_numeric_array(box_phi, name="box_phi")
    if box.shape != field.shape:
        raise ValueError("box_phi must have the same shape as phi")
    curvature = _real_scalar_or_field(
        scalar_curvature, field.shape, name="scalar_curvature"
    )
    xi_value = _finite_scalar(xi, name="xi")
    residual = box + potential.gradient(field) - xi_value * curvature * field
    if not np.all(np.isfinite(residual)):
        raise ValueError("curved field residual must be finite")
    return residual


def einstein_equation_residual(
    einstein_tensor,
    stress_energy,
    gravitational_constant,
) -> np.ndarray:
    """Return ``G_mu_nu - 8*pi*G*T_mu_nu``.

    The two tensors must have the same explicit shape ``(...,4,4)``.  This is
    only an equation evaluator; it does not infer or solve for a metric.
    """

    einstein = _metric_array(einstein_tensor, name="einstein_tensor")
    stress = _metric_array(stress_energy, name="stress_energy")
    if einstein.shape != stress.shape:
        raise ValueError("einstein_tensor and stress_energy must have identical shapes")
    constant = _positive_gravitational_constant(gravitational_constant)
    return einstein - 8.0 * np.pi * constant * stress
