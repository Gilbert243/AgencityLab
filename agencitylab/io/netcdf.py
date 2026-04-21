"""
NetCDF persistence helpers.

This module is optional. It supports xarray first and falls back to netCDF4
when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from .json import _to_jsonable


def _require_xarray_or_netcdf4():
    """Import an available NetCDF backend lazily."""
    try:
        import xarray as xr  # type: ignore
        return ("xarray", xr)
    except Exception:
        pass

    try:
        from netCDF4 import Dataset  # type: ignore
        return ("netCDF4", Dataset)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "NetCDF support requires either xarray or netCDF4. Install the corresponding extra."
        ) from exc


def save_netcdf(data: Any, path: Union[str, Path]) -> Path:
    """
    Save a nested structure to NetCDF.

    The implementation is intentionally conservative and best suited for
    dictionaries of arrays.
    """
    backend_name, backend = _require_xarray_or_netcdf4()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(data)

    if backend_name == "xarray":
        xr = backend
        import xarray as xr  # type: ignore

        data_vars = {}
        coords = {}
        attrs = {}

        for key, value in payload.items():
            if isinstance(value, list):
                arr = np.asarray(value)
                if arr.ndim == 1:
                    coords[key] = arr
                else:
                    data_vars[key] = (tuple(f"dim_{i}" for i in range(arr.ndim)), arr)
            else:
                attrs[key] = value

        dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        dataset.to_netcdf(path)
        return path

    Dataset = backend
    with Dataset(path, "w") as ds:
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(value, list):
                    arr = np.asarray(value)
                    if arr.ndim == 1:
                        ds.createDimension(str(key), arr.shape[0])
                        var = ds.createVariable(str(key), arr.dtype.str, (str(key),))
                        var[:] = arr
                    else:
                        dim_names = []
                        for axis, size in enumerate(arr.shape):
                            dim_name = f"{key}_dim_{axis}"
                            ds.createDimension(dim_name, size)
                            dim_names.append(dim_name)
                        var = ds.createVariable(str(key), arr.dtype.str, tuple(dim_names))
                        var[:] = arr
                else:
                    setattr(ds, str(key), value)
        else:
            setattr(ds, "value", payload)

    return path


def load_netcdf(path: Union[str, Path]) -> Any:
    """
    Load a NetCDF file into a Python structure.
    """
    backend_name, backend = _require_xarray_or_netcdf4()
    path = Path(path)

    if backend_name == "xarray":
        import xarray as xr  # type: ignore
        dataset = xr.load_dataset(path)
        return dataset.to_dict()

    Dataset = backend
    result = {}
    with Dataset(path, "r") as ds:
        for key, var in ds.variables.items():
            result[key] = var[...].tolist()
        for key in ds.ncattrs():
            result[key] = getattr(ds, key)
    return result
