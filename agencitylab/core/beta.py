"""Canonical intrinsic agencity state ``beta = J U``."""

from __future__ import annotations

import numpy as np

from .contrast import compute_contrast
from .orientation import compute_orientation


def _beta_stats(beta_signal):
    magnitude = np.abs(beta_signal)
    return {
        "mean": float(np.mean(magnitude)),
        "std": float(np.std(magnitude)),
        "min": float(np.min(magnitude)),
        "max": float(np.max(magnitude)),
        "nonzero_fraction": float(np.mean(magnitude > 0.0)),
    }


def compute_beta(
    D,
    S,
    M,
    O,
    *,
    contrast_robust=False,
    contrast_scale=10.0,
    beta_clip=None,
    null_threshold=None,
    return_components=False,
    verbose=False,
):
    """Compute canonical ``J``, ``U`` and ``beta``.

    For ``S = 0`` the orientation returned by :func:`compute_orientation` is zero,
    hence beta is exactly zero. No epsilon, saturation, or clipping is introduced.
    """
    del contrast_scale
    if contrast_robust or beta_clip is not None:
        raise ValueError("canonical beta cannot be saturated or clipped")
    if null_threshold not in {None, 0, 0.0}:
        raise ValueError("canonical beta uses exact S = 0, not a threshold")

    J = compute_contrast(D, S, verbose=verbose)
    U, S_internal = compute_orientation(
        M,
        O,
        null_threshold=null_threshold,
        return_intensity=True,
        verbose=verbose,
    )
    supplied_S = np.asarray(S, dtype=float)
    if supplied_S.shape != S_internal.shape or not np.allclose(supplied_S, S_internal, rtol=1e-12, atol=0.0):
        raise ValueError("S must equal sqrt(M^2 + O^2)")

    beta_signal = np.asarray(J * U, dtype=complex)
    stats = _beta_stats(beta_signal)
    if verbose:
        print(f"[beta] |beta| mean={stats['mean']:.6g}")

    if return_components:
        return {"J": J, "U": U, "S": S_internal, "beta": beta_signal, "stats": stats}
    return J, U, beta_signal


def beta(D, S, M, O, **kwargs):
    """Return only the canonical intrinsic agencity state."""
    _, _, out = compute_beta(D, S, M, O, **kwargs)
    return out


def structured_agencity(*args, **kwargs):
    """Compatibility alias for :func:`beta`."""
    return beta(*args, **kwargs)
