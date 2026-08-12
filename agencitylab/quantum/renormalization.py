"""Explicit one-loop coupling term from Chapter 21.

Only the coefficient written in the accepted source is implemented. No higher
orders, running-coupling solver, counterterm scheme, or cutoff prescription is
invented here.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.SPECULATIVE


def one_loop_quartic_beta(mu) -> float:
    """Return the stated leading term ``5 mu^2 / (16 pi^2)``.

    Chapter 21 identifies the dimensionless coupling ``g`` with the quartic
    coefficient ``mu`` and states ``beta(g) = 5 g^2/(16 pi^2) + O(g^3)``.
    This function returns only the explicitly given one-loop term.
    """
    try:
        coupling = float(mu)
    except Exception as exc:
        raise ValueError("mu must be a finite positive scalar") from exc
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("mu must be finite and strictly positive")
    return float(5.0 * coupling**2 / (16.0 * np.pi**2))
