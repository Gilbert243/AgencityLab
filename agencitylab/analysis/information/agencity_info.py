"""
Agencity-information bridge utilities.

These helpers connect the Agencity observable with information-oriented
descriptions.
"""

from __future__ import annotations

import numpy as np

from .shannon import shannon_entropy


def agencity_information_index(b, epsilon: float = 1e-12) -> float:
    """
    Compute a synthetic index linking Agencity to information structure.

    The current version measures the entropy of the normalized absolute
    observable. Low entropy means concentrated activity bursts.
    """
    b = np.asarray(b, dtype=float)
    if b.size == 0:
        return 0.0

    weights = np.abs(b)
    total = float(np.sum(weights))
    if total < epsilon:
        return 0.0

    distribution = weights / total
    return float(shannon_entropy(distribution, base=np.e))
