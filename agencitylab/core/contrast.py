"""Canonical logarithmic contrast for AgencityLab."""

from __future__ import annotations

import numpy as np


def _validate_pair(D, S):
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    if D.shape != S.shape:
        raise ValueError("D and S must share the same shape")
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(S)):
        raise ValueError("D and S must contain only finite values")
    if np.any(D < 0.0) or np.any(S < 0.0):
        raise ValueError("D and S must be non-negative")
    return D, S


def compute_contrast(
    D,
    S,
    *,
    eps=None,
    robust=False,
    robust_scale=10.0,
    clip=None,
    return_ratio=False,
    verbose=False,
):
    """Compute exactly ``J = ln((e + D) / (e + S))``.

    Numerical epsilon, saturation, and clipping are deliberately excluded from
    the canonical equation. Compatibility arguments are rejected if activated.
    """
    del robust_scale
    if eps not in {None, 0, 0.0}:
        raise ValueError("canonical contrast does not add epsilon to e + D or e + S")
    if robust or clip is not None:
        raise ValueError("canonical contrast cannot be saturated or clipped")

    D, S = _validate_pair(D, S)
    ratio = (np.e + D) / (np.e + S)
    J = np.log(ratio)
    if verbose:
        print(f"[contrast] mean={np.mean(J):.6g}")
    return (J, ratio) if return_ratio else J
