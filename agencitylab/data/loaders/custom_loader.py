"""
Custom loader adapter (multi-dimensional support).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def load_custom_signal(
    data: Any,
    extractor: Callable[[Any], Tuple[Any, Any]]
) -> Tuple[np.ndarray, np.ndarray]:

    xi, u = extractor(data)

    xi = np.asarray(xi, dtype=float)
    u = np.asarray(u, dtype=float)

    if xi.ndim != 1:
        raise ValueError("xi must be 1D")

    if u.ndim not in (1, 2):
        raise ValueError("u must be 1D or 2D")

    if xi.shape[0] != u.shape[0]:
        raise ValueError("xi and u must have same length")

    return xi, u