"""
Custom loader adapter.

Use this module when the raw data source has a project-specific structure.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def load_custom_signal(data: Any, extractor: Callable[[Any], Tuple[Any, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a signal with a user-provided extractor function.

    Parameters
    ----------
    data:
        Raw source object.
    extractor:
        Function returning (xi, u).
    """
    xi, u = extractor(data)
    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)

    if xi.ndim != 1 or u.ndim != 1:
        raise ValueError("Extractor must return one-dimensional arrays.")

    if xi.shape[0] != u.shape[0]:
        raise ValueError("Extractor returned arrays of different lengths.")

    return xi, u
