"""
Pandas signal loader (multi-dimensional support).
"""

from __future__ import annotations

from typing import Tuple, Union, List

import numpy as np


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as exc:
        raise ImportError("pandas is required") from exc


def load_pandas_signal(
    data,
    xi_col: str = "xi",
    u_cols: Union[str, List[str]] = "u"
) -> Tuple[np.ndarray, np.ndarray]:

    pd = _require_pandas()

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")

    if xi_col not in data.columns:
        raise ValueError(f"Missing column '{xi_col}'")

    xi = data[xi_col].to_numpy(dtype=float)

    # 🔥 MULTI-D
    if isinstance(u_cols, str):
        if u_cols not in data.columns:
            raise ValueError(f"Missing column '{u_cols}'")
        u = data[u_cols].to_numpy(dtype=float)

    else:
        for col in u_cols:
            if col not in data.columns:
                raise ValueError(f"Missing column '{col}'")
        u = data[u_cols].to_numpy(dtype=float)

    if u.ndim == 2 and u.shape[1] == 1:
        u = u[:, 0]

    return xi, u