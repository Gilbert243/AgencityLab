"""
JSON signal loader.

Accepted formats:
- {"xi": [...], "u": [...]}
- {"data": [{"xi": ..., "u": ...}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np


def load_json_signal(source: Union[str, Path, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load xi and u from a JSON file or dictionary.
    """
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        payload = source

    if isinstance(payload, dict) and "xi" in payload and "u" in payload:
        return np.asarray(payload["xi"], dtype=float), np.asarray(payload["u"], dtype=float)

    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
        xi = [item["xi"] for item in data]
        u = [item["u"] for item in data]
        return np.asarray(xi, dtype=float), np.asarray(u, dtype=float)

    raise ValueError("Unsupported JSON signal format.")
