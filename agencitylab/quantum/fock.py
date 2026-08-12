"""Finite-dimensional Fock-space approximations for Agencity agenton modes.

Chapter 21 states the bosonic creation/annihilation algebra in continuous
momentum space. A finite matrix cannot satisfy the exact canonical commutator
on every state. These helpers therefore expose an explicit numerical
truncation: the commutator is canonical below the highest retained state and
has the unavoidable cutoff defect on that state.
"""

from __future__ import annotations

import operator

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


def _cutoff(value) -> int:
    try:
        cutoff = operator.index(value)
    except Exception as exc:
        raise ValueError("cutoff must be a positive integer") from exc
    if cutoff < 1:
        raise ValueError("cutoff must be at least 1")
    return cutoff


def _occupation(value, *, cutoff: int) -> int:
    try:
        occupation = operator.index(value)
    except Exception as exc:
        raise ValueError("occupation must be an integer") from exc
    if occupation < 0 or occupation >= cutoff:
        raise ValueError("occupation must satisfy 0 <= occupation < cutoff")
    return occupation


def annihilation_operator(cutoff) -> np.ndarray:
    """Return the truncated bosonic annihilation matrix in the number basis."""
    size = _cutoff(cutoff)
    matrix = np.zeros((size, size), dtype=complex)
    for n in range(1, size):
        matrix[n - 1, n] = np.sqrt(n)
    return matrix


def creation_operator(cutoff) -> np.ndarray:
    """Return the Hermitian adjoint of the truncated annihilation matrix."""
    return annihilation_operator(cutoff).conj().T


def number_operator(cutoff) -> np.ndarray:
    """Return the truncated number operator ``a^dagger a``."""
    size = _cutoff(cutoff)
    return np.diag(np.arange(size, dtype=float)).astype(complex)


def fock_state(occupation, cutoff) -> np.ndarray:
    """Return the retained number-basis vector ``|occupation>``."""
    size = _cutoff(cutoff)
    n = _occupation(occupation, cutoff=size)
    state = np.zeros(size, dtype=complex)
    state[n] = 1.0
    return state


def vacuum_state(cutoff) -> np.ndarray:
    """Return the truncated Fock vacuum ``|0>``."""
    return fock_state(0, cutoff)


def occupation_expectation(state) -> float:
    """Return the number expectation as a norm-independent Rayleigh quotient."""
    vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("state must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("state must contain only finite values")
    norm_squared = float(np.real(np.vdot(vector, vector)))
    if norm_squared <= 0.0:
        raise ValueError("state must be non-zero")
    number = number_operator(vector.size)
    value = np.vdot(vector, number @ vector) / norm_squared
    value = np.real_if_close(value, tol=1000)
    if np.iscomplexobj(value):
        raise ValueError("occupation expectation must be theoretically real")
    return float(value)


def truncated_commutator(cutoff) -> np.ndarray:
    """Return ``[a, a^dagger]`` for the finite retained basis."""
    annihilation = annihilation_operator(cutoff)
    creation = annihilation.conj().T
    return annihilation @ creation - creation @ annihilation


def truncation_commutator_defect(cutoff) -> np.ndarray:
    """Return the finite-cutoff defect ``[a,a^dagger] - I``.

    For basis size ``N`` the exact matrix is ``-N |N-1><N-1|``. Exposing this
    quantity prevents a finite numerical approximation from being mistaken for
    the exact infinite-dimensional canonical algebra.
    """
    size = _cutoff(cutoff)
    return truncated_commutator(size) - np.eye(size, dtype=complex)
