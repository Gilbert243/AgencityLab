"""
High-level save helpers.

These helpers dispatch to JSON, HDF5 or NetCDF depending on the file
extension or the explicit format argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .hdf5 import save_hdf5
from .json import dump_json
from .netcdf import save_netcdf


def save(data: Any, path: Union[str, Path], format: Optional[str] = None) -> Path:
    """
    Save data to disk using a format inferred from the path or supplied
    explicitly.

    Supported formats: json, hdf5, netcdf.
    """
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt in {"json", "js"}:
        return dump_json(data, path)
    if fmt in {"h5", "hdf5"}:
        return save_hdf5(data, path)
    if fmt in {"nc", "netcdf"}:
        return save_netcdf(data, path)

    raise ValueError(f"Unsupported save format: {fmt}")


def save_to_path(data: Any, path: Union[str, Path], format: Optional[str] = None) -> Path:
    """Alias of save() kept for readability in higher-level APIs."""
    return save(data, path, format=format)
