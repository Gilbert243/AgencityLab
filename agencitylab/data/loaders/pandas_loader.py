"""
Pandas loader for tabular signal data.

Pandas is optional. The function raises a clear ImportError if pandas is
not installed.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _require_pandas():
    """Import pandas lazily."""
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pandas is required for this loader. Install AgencityLab with the pandas extra."
        ) from exc


def load_pandas_signal(data: Any, xi_col: str = "xi", u_col: str = "u") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a signal from a pandas DataFrame.
    """
    pd = _require_pandas()

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if xi_col not in data.columns or u_col not in data.columns:
        raise ValueError(f"DataFrame must contain '{xi_col}' and '{u_col}' columns.")

    xi = data[xi_col].to_numpy(dtype=float)
    u = data[u_col].to_numpy(dtype=float)
    return xi, u
