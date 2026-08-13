"""Analytical robustness relations stated in Volume 2 Chapters 4 and 10.

These helpers are analysis-side mathematical consequences.  They do not make
``e`` a tunable canonical parameter and do not modify characteristic power.
"""

from __future__ import annotations

import numpy as np


def logarithmic_contrast_offset_sensitivity(D, S):
    """Return the Chapter-4 sensitivity ``partial J / partial e`` at ``e=exp(1)``.

    The canonical contrast remains ``J = log((e + D)/(e + S))`` with the fixed
    mathematical constant ``e = exp(1)``.  This function only evaluates the
    derivative used by the theory's sensitivity analysis:

    ``(S - D) / ((e + D) * (e + S))``.
    """
    D_arr, S_arr = np.broadcast_arrays(
        np.asarray(D, dtype=float),
        np.asarray(S, dtype=float),
    )
    if not np.all(np.isfinite(D_arr)) or not np.all(np.isfinite(S_arr)):
        raise ValueError("D and S must contain only finite values")
    if np.any(D_arr < 0.0) or np.any(S_arr < 0.0):
        raise ValueError("D and S must be non-negative")
    out = (S_arr - D_arr) / ((np.e + D_arr) * (np.e + S_arr))
    return out.item() if out.ndim == 0 else out


def multiplicative_power_perturbation(b0, relative_error):
    """Apply the Chapter-10 multiplicative characteristic-power perturbation.

    If ``P_c = P_c0 * (1 + epsilon)``, the theory gives exactly
    ``b = b0 + epsilon*b0``.  The returned pair is ``(b, delta_b)`` with
    ``delta_b = epsilon*b0``.  No stochastic model or error distribution is
    assumed here.
    """
    b_arr, error_arr = np.broadcast_arrays(
        np.asarray(b0, dtype=complex),
        np.asarray(relative_error, dtype=float),
    )
    if not np.all(np.isfinite(b_arr)):
        raise ValueError("b0 must contain only finite values")
    if not np.all(np.isfinite(error_arr)):
        raise ValueError("relative_error must contain only finite values")
    delta = error_arr * b_arr
    perturbed = b_arr + delta
    if perturbed.ndim == 0:
        return perturbed.item(), delta.item()
    return perturbed, delta


__all__ = [
    "logarithmic_contrast_offset_sensitivity",
    "multiplicative_power_perturbation",
]
