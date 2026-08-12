"""Appendix-B field-theoretic formulation written directly in terms of ``beta``.

Volume 2 Appendix B separately records the flat-field action

    L_beta = 1/2 P_c^2 partial_mu beta partial^mu conjugate(beta) - V(|beta|),

the equation ``box(beta) + gradient(V)/P_c^2 = 0``, the corresponding
energy--momentum tensor, and the U(1) current
``J^mu = P_c^2 Im(beta partial^mu conjugate(beta))``.

This source-layer formulation is intentionally kept distinct from the
Chapter-15 bridge ``phi = sqrt(P_c tau) beta`` and the Chapter-16 canonically
normalised ``phi`` theory.  AgencityLab does not silently claim that the two
normalisations are algebraically identical.  ``P_c`` is therefore required to
be a finite strictly positive scalar here when it appears in a denominator;
this research-only restriction does not alter the canonical observable rule
that ``P_c = 0`` is valid and gives ``b = 0``.

Scientific status: research.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from .conservation import (
    _field_and_derivatives,
    _raised_derivatives,
    _real_if_close,
    flat_field_minkowski_metric,
)
from .potential import QuarticAgencityPotential

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _positive_characteristic_power(value) -> float:
    try:
        power = float(value)
    except Exception as exc:
        raise ValueError("P_c must be a finite strictly positive scalar") from exc
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError(
            "P_c must be finite and strictly positive in the Appendix-B beta-field formulation"
        )
    return power


def appendix_b_beta_lagrangian_density(
    beta,
    derivatives,
    potential: QuarticAgencityPotential,
    *,
    P_c: float,
) -> np.ndarray:
    """Return the Appendix-B flat beta-field Lagrangian density."""

    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    power = _positive_characteristic_power(P_c)
    field, deriv = _field_and_derivatives(beta, derivatives)
    raised = _raised_derivatives(deriv)
    contraction = np.sum(np.conjugate(deriv) * raised, axis=-1)
    lagrangian = 0.5 * power**2 * contraction - potential.value(field)
    return _real_if_close(lagrangian, name="Appendix-B beta-field Lagrangian density")


def appendix_b_beta_equation_residual(
    beta,
    box_beta,
    potential: QuarticAgencityPotential,
    *,
    P_c: float,
) -> np.ndarray:
    """Evaluate the Appendix-B beta equation of motion.

    The returned residual is
    ``box(beta) + potential.gradient(beta) / P_c**2``.
    """

    if not isinstance(potential, QuarticAgencityPotential):
        raise TypeError("potential must be a QuarticAgencityPotential")
    power = _positive_characteristic_power(P_c)
    field = np.asarray(beta)
    box = np.asarray(box_beta)
    if field.shape != box.shape or field.size == 0:
        raise ValueError("beta and box_beta must be non-empty arrays with identical shape")
    if not np.issubdtype(field.dtype, np.number) or not np.issubdtype(box.dtype, np.number):
        raise TypeError("beta and box_beta must contain numeric values")
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(box)):
        raise ValueError("beta and box_beta must contain only finite values")
    return box + potential.gradient(field) / power**2


def appendix_b_beta_noether_current(beta, derivatives, *, P_c: float) -> np.ndarray:
    """Return the Appendix-B U(1) current ``P_c^2 Im(beta d^mu beta*)``."""

    power = _positive_characteristic_power(P_c)
    field, deriv = _field_and_derivatives(beta, derivatives)
    raised = _raised_derivatives(deriv)
    return power**2 * np.imag(field[..., np.newaxis] * np.conjugate(raised))


def appendix_b_beta_energy_momentum_tensor(
    beta,
    derivatives,
    potential: QuarticAgencityPotential,
    *,
    P_c: float,
) -> np.ndarray:
    """Return the Appendix-B flat beta-field energy--momentum tensor."""

    power = _positive_characteristic_power(P_c)
    field, deriv = _field_and_derivatives(beta, derivatives)
    raised = _raised_derivatives(deriv)
    lagrangian = appendix_b_beta_lagrangian_density(
        field,
        deriv,
        potential,
        P_c=power,
    )
    first = raised[..., :, np.newaxis] * np.conjugate(raised[..., np.newaxis, :])
    second = raised[..., np.newaxis, :] * np.conjugate(raised[..., :, np.newaxis])
    tensor = 0.5 * power**2 * (first + second)
    tensor = tensor - flat_field_minkowski_metric() * lagrangian[..., np.newaxis, np.newaxis]
    tensor = _real_if_close(tensor, name="Appendix-B beta-field energy-momentum tensor")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("Appendix-B beta-field energy-momentum tensor must be finite")
    return tensor
