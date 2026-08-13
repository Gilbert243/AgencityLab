"""Energy-balance residual for dissipative Agencity fields.

The source equation is Volume 2, Eq. (18.3):
``partial_t H + div(J_E) = -Gamma * |partial_t phi|**2``.
The manuscript does not define a unique numerical energy-flux discretisation
here, so this module evaluates the balance from explicit supplied terms rather
than inventing ``J_E``.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

from .dissipation import dissipation_density

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_real_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def energy_balance_residual(dH_dt, div_j_e, phi_dot, gamma: float) -> np.ndarray:
    """Evaluate ``dH_dt + div(J_E) + Gamma*|phi_dot|**2``.

    A zero result is the numerical statement of Volume 2 Eq. (18.3). The
    residual is not clipped or forced to zero.
    """
    energy_rate = _finite_real_array(dH_dt, name="dH_dt")
    flux_divergence = _finite_real_array(div_j_e, name="div_j_e")
    dissipated = dissipation_density(phi_dot, gamma)
    try:
        energy_b, flux_b, dissipated_b = np.broadcast_arrays(
            energy_rate,
            flux_divergence,
            dissipated,
        )
    except ValueError as exc:
        raise ValueError(
            "dH_dt, div_j_e, and phi_dot are not broadcast-compatible"
        ) from exc
    return np.asarray(energy_b + flux_b + dissipated_b, dtype=float)
