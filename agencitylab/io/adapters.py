"""Optional tabular adapters for Agencity result objects."""

from __future__ import annotations

from typing import Any

import numpy as np


def result_to_dataframe(result: Any):
    """Convert a result to a pandas DataFrame when pandas is installed."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for to_dataframe(); install agencitylab[data]") from exc

    power = (
        result.P_c
        if result.is_time_varying_power
        else np.full(len(result), float(result.P_c), dtype=float)
    )
    frame = pd.DataFrame(
        {
            "xi": result.xi,
            "u": result.u,
            "u_star": result.u_star,
            "X_star": result.X_star,
            "A_star": result.A_star,
            "M": result.M,
            "O": result.O,
            "D": result.D,
            "S": result.S,
            "J": result.J,
            "theta": result.theta,
            "P_c": power,
            "beta_real": result.beta.real,
            "beta_imag": result.beta.imag,
            "beta_abs": result.beta_abs,
            "b_real": result.b.real,
            "b_imag": result.b.imag,
            "b_abs": result.b_abs,
        }
    )
    frame.attrs["units"] = result.metadata.unit_contract()
    return frame


def result_to_xarray(result: Any, *, schema_version: str):
    """Convert a result to an xarray Dataset when xarray is installed."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("xarray is required for to_xarray(); install agencitylab[data]") from exc

    data_vars = {
        "u": ("xi", result.u),
        "u_star": ("xi", result.u_star),
        "X_star": ("xi", result.X_star),
        "A_star": ("xi", result.A_star),
        "M": ("xi", result.M),
        "O": ("xi", result.O),
        "D": ("xi", result.D),
        "S": ("xi", result.S),
        "J": ("xi", result.J),
        "theta": ("xi", result.theta),
        "beta_abs": ("xi", result.beta_abs),
        "b_abs": ("xi", result.b_abs),
    }
    attrs = {
        "schema_version": schema_version,
        "tau": result.tau,
        "A_ref": result.A_ref,
        "memory_window": result.memory_window,
        "unit": result.unit,
        "coordinate_unit": result.coordinate_unit,
        "power_unit": result.power_unit,
        "b_unit": result.b_unit,
    }
    if result.is_time_varying_power:
        data_vars["P_c"] = ("xi", result.P_c)
    else:
        attrs["P_c"] = float(result.P_c)

    dataset = xr.Dataset(data_vars=data_vars, coords={"xi": result.xi}, attrs=attrs)
    dataset["xi"].attrs["unit"] = result.coordinate_unit
    dataset["u"].attrs["unit"] = result.unit
    dataset["b_abs"].attrs["unit"] = result.b_unit
    if "P_c" in dataset:
        dataset["P_c"].attrs["unit"] = result.power_unit
    return dataset
