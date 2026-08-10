"""
CSV signal loader (multi-dimensional support).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, List

import numpy as np


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as exc:
        raise ImportError(
            "pandas is required for CSV loading."
        ) from exc


def load_csv_signal(
    path: Union[str, Path],
    xi_col: str = "xi",
    u_cols: Union[str, List[str]] = "u",
) -> Tuple[np.ndarray, np.ndarray]:

    pd = _require_pandas()
    df = pd.read_csv(Path(path))

    if xi_col not in df.columns:
        raise ValueError(f"Missing column '{xi_col}'")

    xi = df[xi_col].to_numpy(dtype=float)

    # 🔥 MULTI-D SUPPORT
    if isinstance(u_cols, str):
        if u_cols not in df.columns:
            raise ValueError(f"Missing column '{u_cols}'")
        u = df[u_cols].to_numpy(dtype=float)

    else:
        for col in u_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'")
        u = df[u_cols].to_numpy(dtype=float)

    # 🔥 normalisation shape
    if u.ndim == 2 and u.shape[1] == 1:
        u = u[:, 0]

    return xi, u