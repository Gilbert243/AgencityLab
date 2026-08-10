"""
High-level load helpers.

These helpers dispatch to JSON, CSV, HDF5 or NetCDF depending on the file
extension or the explicit format argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .csv import load_csv
from .hdf5 import load_hdf5
from .json import load_json
from .netcdf import load_netcdf


def load(path: Union[str, Path], format: Optional[str] = None) -> Any:
    """
    Load data from disk using a format inferred from the path or supplied
    explicitly.

    Supported formats: json, csv, hdf5, netcdf.
    """
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt in {"json", "js"}:
        return load_json(path)
    if fmt in {"csv"}:
        return load_csv(path)
    if fmt in {"h5", "hdf5"}:
        return load_hdf5(path)
    if fmt in {"nc", "netcdf"}:
        return load_netcdf(path)

    raise ValueError(f"Unsupported load format: {fmt}")


def load_from_path(path: Union[str, Path], format: Optional[str] = None) -> Any:
    """
    Alias of load() kept for readability in higher-level APIs.
    """
    return load(path, format=format)