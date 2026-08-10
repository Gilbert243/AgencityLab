"""
JSON signal loader (multi-dimensional support).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def load_json_signal(source: Union[str, Path, dict]) -> Tuple[np.ndarray, np.ndarray]:

    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source

    # =========================
    # format direct
    # =========================
    if isinstance(payload, dict) and "xi" in payload and "u" in payload:
        xi = np.asarray(payload["xi"], dtype=float)
        u = np.asarray(payload["u"], dtype=float)

    # =========================
    # format liste
    # =========================
    elif isinstance(payload, dict) and "data" in payload:
        data = payload["data"]

        xi = np.asarray([item["xi"] for item in data], dtype=float)
        u_list = [item["u"] for item in data]
        u = np.asarray(u_list, dtype=float)

    else:
        raise ValueError("Unsupported JSON format")

    if xi.ndim != 1:
        raise ValueError("xi must be 1D")

    if u.ndim not in (1, 2):
        raise ValueError("u must be 1D or 2D")

    if xi.shape[0] != u.shape[0]:
        raise ValueError("xi and u length mismatch")

    return xi, u