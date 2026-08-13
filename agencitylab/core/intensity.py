"""Canonical dynamic and structural intensities for AgencityLab."""

from __future__ import annotations

import numpy as np


def _validate_pair(a, b, name_a, name_b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must share the same shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{name_a} and {name_b} must contain only finite values")
    return a, b


def compute_dynamic_intensity(
    X,
    A,
    *,
    clip=None,
    robust=False,
    robust_scale=10.0,
    return_components=False,
    verbose=False,
):
    """Compute exactly ``D = sqrt(X^2 + (A X)^2)``."""
    del robust_scale
    if clip is not None or robust:
        raise ValueError("canonical dynamic intensity cannot be clipped or saturated")
    X, A = _validate_pair(X, A, "X", "A")
    AX = A * X
    D = np.hypot(X, AX)
    if verbose:
        print(f"[intensity] D mean={np.mean(D):.6g}")
    return (D, AX) if return_components else D


def compute_structural_intensity(
    M,
    O,
    *,
    null_threshold=None,
    return_mask=False,
    verbose=False,
):
    """Compute exactly ``S = sqrt(M^2 + O^2)``."""
    if null_threshold not in {None, 0, 0.0}:
        raise ValueError("canonical structural intensity uses exact S = 0, not a threshold")
    M, O = _validate_pair(M, O, "M", "O")
    S = np.hypot(M, O)
    mask = S > 0.0
    if verbose:
        print(f"[intensity] S mean={np.mean(S):.6g}")
    return (S, mask) if return_mask else S


def compute_intensities(
    X,
    A,
    M,
    O,
    *,
    dynamic_clip=None,
    robust_dynamic=False,
    robust_scale=10.0,
    structural_threshold=None,
    verbose=False,
):
    """Compute canonical ``D`` and ``S``."""
    D = compute_dynamic_intensity(
        X,
        A,
        clip=dynamic_clip,
        robust=robust_dynamic,
        robust_scale=robust_scale,
        verbose=verbose,
    )
    S = compute_structural_intensity(
        M,
        O,
        null_threshold=structural_threshold,
        verbose=verbose,
    )
    return D, S
