"""Canonical structural orientation for AgencityLab."""

from __future__ import annotations

import numpy as np


def compute_orientation(
    M,
    O,
    *,
    null_threshold=None,
    return_intensity=False,
    return_mask=False,
    verbose=False,
):
    """Compute ``U = (M + i O) / S`` for ``S > 0`` and ``U = 0`` for ``S = 0``."""
    if null_threshold not in {None, 0, 0.0}:
        raise ValueError("canonical orientation uses exact S = 0, not a threshold")

    M = np.asarray(M, dtype=float)
    O = np.asarray(O, dtype=float)
    if M.shape != O.shape:
        raise ValueError("M and O must share the same shape")
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(O)):
        raise ValueError("M and O must contain only finite values")

    S = np.hypot(M, O)
    valid = S > 0.0
    U = np.zeros(S.shape, dtype=complex)
    U[valid] = (M[valid] + 1j * O[valid]) / S[valid]

    if verbose:
        print(f"[orientation] valid_fraction={np.mean(valid):.6g}")

    if return_intensity and return_mask:
        return U, S, valid
    if return_intensity:
        return U, S
    if return_mask:
        return U, valid
    return U


def compute_angle(b, *, degrees=False, unwrap=False, verbose=False):
    """Return the argument of a complex field."""
    theta = np.angle(np.asarray(b))
    if unwrap:
        theta = np.unwrap(theta)
    if degrees:
        theta = np.degrees(theta)
    if verbose:
        print(f"[angle] mean={np.mean(theta):.6g}")
    return theta
