"""
Agencity-information bridge utilities.

This module connects:
    - agencity observable b(t)
    - structural measures (J, beta, theta)
    - information measures
"""

from __future__ import annotations
import numpy as np

from .shannon import shannon_entropy_from_signal

EPS = 1e-12


def agencity_information_index(b, *, verbose=False) -> float:
    """
    Information content of agencity signal.

    Defined as entropy of normalized |b|.
    """
    b = np.asarray(b)
    mag = np.abs(b)

    H = shannon_entropy_from_signal(mag)

    if verbose:
        print(f"[info] entropy(|b|) = {H:.6f}")

    return H


def agencity_information_density(b, *, verbose=False) -> float:
    """
    Information density:
        entropy / variability
    """
    b = np.asarray(b)
    mag = np.abs(b)

    H = shannon_entropy_from_signal(mag)
    std = np.std(mag)

    density = H / (std + EPS)

    if verbose:
        print(f"[info] density = {density:.6f}")

    return float(density)


def agencity_structural_information(J, *, verbose=False):
    """
    Structural information via contrast J.

    Measures asymmetry structure.
    """
    J = np.asarray(J, dtype=float)

    info = np.mean(np.abs(J))

    if verbose:
        print(f"[info] structural J = {info:.6f}")

    return float(info)


def agencity_phase_information(theta, *, verbose=False):
    """
    Information contained in orientation dynamics.
    """
    theta = np.asarray(theta, dtype=float)

    H = shannon_entropy_from_signal(theta)

    if verbose:
        print(f"[info] phase entropy = {H:.6f}")

    return H


def full_information_summary(b, J=None, theta=None, *, verbose=False):
    """Return the descriptive information summary without altering canonical data."""
    out = {
        "entropy_b": agencity_information_index(b, verbose=verbose),
        "density_b": agencity_information_density(b, verbose=verbose),
    }

    if J is not None:
        out["structure_J"] = agencity_structural_information(J, verbose=verbose)

    if theta is not None:
        out["phase_entropy"] = agencity_phase_information(theta, verbose=verbose)

    return out
