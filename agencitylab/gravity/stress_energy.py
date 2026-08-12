"""Stress-energy primitives for the Chapter-19 Agencity gravity model.

Volume 2 gives the complete field stress-energy tensor only for minimal
coupling ``xi = 0``.  For ``xi != 0`` it states only that additional Einstein-
tensor and second-derivative terms occur, without providing their full formula.
This module therefore refuses nonminimal stress-energy evaluation rather than
importing an external convention and presenting it as source theory.

Scientific status: research.
"""

from __future__ import annotations

import numpy as np

from agencitylab.fields.physics import QuarticAgencityPotential
from agencitylab.scientific_status import ScientificStatus

from .action import _finite_scalar, matter_lagrangian_density
from .geometry import (
    _finite_numeric_array,
    _metric_for_shape,
    _real_if_numerically_close,
)

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def stress_energy_tensor(
    phi,
    derivatives,
    metric,
    inverse_metric,
    potential: QuarticAgencityPotential,
    *,
    xi=0.0,
) -> np.ndarray:
    """Return the complete Chapter-19 minimal stress-energy tensor.

    For ``xi = 0`` the source formula is

    ``T_mu_nu = nabla_(mu) bar(phi) nabla_(nu) phi
                 - g_mu_nu [1/2 nabla_a bar(phi) nabla^a phi - V]``.

    Parentheses denote symmetrisation, implemented explicitly as the average of
    the two derivative products.  Complex scalar fields are supported.

    ``derivatives`` must already be the derivatives intended by the caller.  If
    the optional Chapter-23 external gauge notation is desired, callers may pass
    the output of :func:`agencitylab.gravity.covariant_scalar_derivative`.
    This tensor still excludes any stress-energy contribution of the external
    gauge field itself.

    Nonminimal ``xi != 0`` raises :class:`NotImplementedError` because the
    accepted source document does not specify the complete tensor.
    """

    xi_value = _finite_scalar(xi, name="xi")
    if xi_value != 0.0:
        raise NotImplementedError(
            "Volume 2 specifies only the structure, not the complete formula, "
            "of the nonminimal xi != 0 stress-energy tensor"
        )
    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")

    field = _finite_numeric_array(phi, name="phi")
    derivative_array = _finite_numeric_array(derivatives, name="derivatives")
    expected = field.shape + (4,)
    if derivative_array.shape != expected:
        raise ValueError(f"derivatives must have shape {expected}")

    covariant_metric = _metric_for_shape(metric, field.shape, name="metric")
    inverse = _metric_for_shape(inverse_metric, field.shape, name="inverse_metric")
    lagrangian = matter_lagrangian_density(field, derivative_array, inverse, potential)

    first = np.conjugate(derivative_array)[..., :, np.newaxis] * derivative_array[
        ..., np.newaxis, :
    ]
    second = np.conjugate(derivative_array)[..., np.newaxis, :] * derivative_array[
        ..., :, np.newaxis
    ]
    symmetrized = 0.5 * (first + second)
    tensor = symmetrized - covariant_metric * lagrangian[..., np.newaxis, np.newaxis]
    tensor = _real_if_numerically_close(tensor, name="minimal stress-energy tensor")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("stress-energy tensor must be finite")
    return tensor
