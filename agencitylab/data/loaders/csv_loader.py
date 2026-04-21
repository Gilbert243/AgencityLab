"""
CSV signal loader.

This module supports standard CSV files with at least two columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np


def _require_pandas():
    """Import pandas lazily for CSV parsing."""
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pandas is required for CSV loading. Install AgencityLab with the pandas extra."
        ) from exc


def load_csv_signal(path: Union[str, Path], xi_col: str = "xi", u_col: str = "u") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load xi and u from a CSV file.

    The CSV is expected to contain at least the columns named xi_col and u_col.
    """
    pd = _require_pandas()
    path = Path(path)
    df = pd.read_csv(path)
    if xi_col not in df.columns or u_col not in df.columns:
        raise ValueError(f"CSV file must contain '{xi_col}' and '{u_col}' columns.")
    return df[xi_col].to_numpy(dtype=float), df[u_col].to_numpy(dtype=float)
