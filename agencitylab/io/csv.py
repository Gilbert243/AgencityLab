"""
CSV serialization helpers for AgencityLab.

Supports:
- flat dictionaries
- nested dictionaries (flattened with dotted keys)
- lists of dictionaries
- dataclasses and objects exposing to_dict()
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_plain(value.to_dict())

    if is_dataclass(value):
        return _to_plain(asdict(value))

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, complex):
        return f"{value.real}+{value.imag}j"

    if isinstance(value, np.complexfloating):
        c = complex(value)
        return f"{c.real}+{c.imag}j"

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]

    return value


def _flatten_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        v = _to_plain(v)
        if isinstance(v, Mapping):
            items.update(_flatten_dict(v, key, sep=sep))
        elif isinstance(v, list) and v and all(isinstance(x, Mapping) for x in v):
            for i, item in enumerate(v):
                items.update(_flatten_dict(item, f"{key}{sep}{i}", sep=sep))
        else:
            items[key] = v
    return items


def _normalize_rows(data: Any) -> List[Dict[str, Any]]:
    """
    Convert input to a list of flat rows.
    """
    data = _to_plain(data)

    if isinstance(data, Mapping):
        return [_flatten_dict(data)]

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        rows: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, Mapping):
                rows.append(_flatten_dict(item))
            else:
                rows.append({"value": _to_plain(item)})
        return rows

    return [{"value": data}]


def dump_csv(data: Any, path: Union[str, Path]) -> Path:
    """
    Write a dictionary or list of dictionaries to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = _normalize_rows(data)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return path


def load_csv(path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Read a CSV file into a list of dictionaries.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]