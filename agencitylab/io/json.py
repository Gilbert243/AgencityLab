"""
JSON serialization helpers.

The functions in this module support plain Python objects, dataclasses,
NumPy arrays, and objects exposing a to_dict() method.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Recursively convert a Python object into a JSON-serializable form."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())

    if is_dataclass(value):
        return _to_jsonable(asdict(value))

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    return value


def dump_json(data: Any, path: Union[str, Path], *, indent: int = 2, sort_keys: bool = True) -> Path:
    """
    Write an object to a JSON file.

    Parameters
    ----------
    data:
        Data to serialize.
    path:
        Output file path.
    indent:
        JSON indentation level.
    sort_keys:
        Whether to sort object keys.

    Returns
    -------
    Path
        The path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = _to_jsonable(data)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=sort_keys), encoding="utf-8")
    return path


def load_json(path: Union[str, Path]) -> Any:
    """
    Read a JSON file and return the decoded Python object.
    """
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
