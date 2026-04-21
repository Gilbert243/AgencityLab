"""Characteristic power utilities.

The theory allows several practical conventions for P_c. The core exposes
a small set of transparent helpers instead of imposing one physical model.
"""

from __future__ import annotations

import numpy as np

from .safeguards import ensure_positive


def characteristic_power(
    value=None,
    *,
    reference_energy=None,
    tau=None,
    default: float = 1.0,
):
    """Return a characteristic power value.

    Supported conventions
    ---------------------
    - explicit value: returns the value as-is
    - energy/time estimate: reference_energy / tau
    - fallback default: 1.0 by default
    """
    if value is not None:
        return ensure_positive(value)
    if reference_energy is not None and tau is not None:
        return ensure_positive(np.asarray(reference_energy, dtype=float) / np.asarray(tau, dtype=float))
    return ensure_positive(default)


def estimate_characteristic_power(signal, *, tau, method: str = "rms", scale=None):
    """Heuristic power estimate for practical workflows.

    This helper is intentionally simple and domain-agnostic. It can be
    replaced by a physically grounded estimator at higher levels.
    """
    x = np.asarray(signal, dtype=float)
    tau = ensure_positive(tau)

    if scale is None:
        if method == "rms":
            scale = np.sqrt(np.nanmean(np.square(x)))
        elif method == "variance":
            scale = np.sqrt(np.nanvar(x))
        elif method == "amplitude":
            scale = np.nanmax(x) - np.nanmin(x)
        else:
            raise ValueError("Unknown power estimation method")

    scale = ensure_positive(scale)
    return ensure_positive((scale ** 2) / tau)
