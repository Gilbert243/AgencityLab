"""
HDF5 persistence helpers.

This module is optional. It only works when h5py is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from .json import _to_jsonable


def _require_h5py():
    """Import h5py lazily so the base installation stays lightweight."""
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "h5py is required for HDF5 support. Install AgencityLab with the hdf5 extra."
        ) from exc
    return h5py


def save_hdf5(data: Any, path: Union[str, Path]) -> Path:
    """
    Save nested data structures to an HDF5 file.

    Dictionaries become groups, arrays become datasets, and scalar values are
    stored as attributes when appropriate.
    """
    h5py = _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_group(group, obj):
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                key = str(key)
                if isinstance(value, Mapping):
                    subgroup = group.create_group(key)
                    write_group(subgroup, value)
                else:
                    array_value = np.asarray(value)
                    if array_value.dtype.kind in {"U", "O"}:
                        group.attrs[key] = _to_jsonable(value)
                    else:
                        group.create_dataset(key, data=array_value)
        else:
            group.attrs["value"] = _to_jsonable(obj)

    with h5py.File(path, "w") as h5file:
        write_group(h5file, _to_jsonable(data))

    return path


def load_hdf5(path: Union[str, Path]) -> Any:
    """
    Load an HDF5 file into nested Python containers.
    """
    h5py = _require_h5py()
    path = Path(path)

    def read_group(group):
        result = {}
        for key, value in group.items():
            if hasattr(value, "items"):
                result[key] = read_group(value)
            else:
                result[key] = value[...].tolist()
        for key, value in group.attrs.items():
            result[key] = value.tolist() if hasattr(value, "tolist") else value
        return result

    with h5py.File(path, "r") as h5file:
        return read_group(h5file)
