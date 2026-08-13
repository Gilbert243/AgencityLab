"""Stable canonical result model returned by the AgencityLab computational API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metadata import ExperimentMetadata

RESULT_SCHEMA_VERSION = "1.0"


def _positive_scalar(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a scalar number") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite")
    return out


def _nonnegative_power(value: Any, *, n: int) -> float | NDArray[np.float64]:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError("P_c must be numeric") from exc
    if arr.ndim == 0:
        scalar = float(arr)
        if not np.isfinite(scalar) or scalar < 0.0:
            raise ValueError("P_c must be non-negative and finite")
        return scalar
    if arr.ndim != 1 or arr.size != n:
        raise ValueError("time-varying P_c must be one-dimensional and match xi")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("P_c must contain only non-negative finite values")
    return arr


def _finite_1d(value: Any, *, name: str, dtype: Any = None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


@dataclass(slots=True)
class AgencityResult:
    """Canonical scalar-signal result with a stable 1.0 data contract."""

    xi: np.ndarray
    u: np.ndarray
    u_star: np.ndarray
    X_star: np.ndarray
    A_star: np.ndarray
    t_star: np.ndarray
    tau: float
    P_c: Any
    A_ref: float
    M: np.ndarray
    O: np.ndarray
    D: np.ndarray
    S: np.ndarray
    J: np.ndarray
    U: np.ndarray
    beta: np.ndarray
    b: np.ndarray

    theta: np.ndarray | None = None
    unit: str = ""
    coordinate_unit: str = ""
    power_unit: str = ""
    observable_kind: str = ""
    domain: str = ""
    system_type: str = ""
    mechanism: str = ""
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        self.xi = _finite_1d(self.xi, name="xi", dtype=float)
        if self.xi.size < 3:
            raise ValueError("xi must contain at least three samples")
        if np.any(np.diff(self.xi) <= 0.0):
            raise ValueError("xi must be strictly increasing")

        n = self.xi.size
        for name in (
            "u",
            "u_star",
            "X_star",
            "A_star",
            "t_star",
            "M",
            "O",
            "D",
            "S",
            "J",
        ):
            arr = _finite_1d(getattr(self, name), name=name, dtype=float)
            if arr.size != n:
                raise ValueError(f"{name} length must match xi")
            setattr(self, name, arr)

        for name in ("U", "beta", "b"):
            arr = _finite_1d(getattr(self, name), name=name, dtype=complex)
            if arr.size != n:
                raise ValueError(f"{name} length must match xi")
            setattr(self, name, arr)

        self.tau = _positive_scalar(self.tau, name="tau")
        self.P_c = _nonnegative_power(self.P_c, n=n)
        self.A_ref = _positive_scalar(self.A_ref, name="A_ref")

        self.metadata = ExperimentMetadata.from_dict(self.metadata)
        for name in (
            "unit",
            "coordinate_unit",
            "power_unit",
            "observable_kind",
            "domain",
            "system_type",
            "mechanism",
        ):
            value = getattr(self, name)
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            setattr(self, name, value.strip())

        self._synchronize_metadata()

        if self.theta is None:
            self.theta = np.angle(self.U)
        else:
            theta = _finite_1d(self.theta, name="theta", dtype=float)
            if theta.size != n:
                raise ValueError("theta length must match xi")
            self.theta = theta

    def _synchronize_metadata(self) -> None:
        pairs = (
            ("unit", "unit"),
            ("coordinate_unit", "coordinate_unit"),
            ("power_unit", "power_unit"),
            ("observable_kind", "observable_kind"),
            ("domain", "domain"),
            ("system_type", "system_type"),
            ("mechanism", "mechanism"),
        )
        for result_name, metadata_name in pairs:
            result_value = getattr(self, result_name)
            metadata_value = getattr(self.metadata, metadata_name)
            if result_value and metadata_value and result_value != metadata_value:
                raise ValueError(f"{result_name} conflicts with metadata.{metadata_name}")
            if not result_value and metadata_value:
                setattr(self, result_name, metadata_value)
            elif result_value and not metadata_value:
                setattr(self.metadata, metadata_name, result_value)

        physical_pairs = (
            ("reference_amplitude", self.A_ref),
            ("characteristic_time", self.tau),
        )
        for name, value in physical_pairs:
            current = getattr(self.metadata, name)
            if current is not None and current != value:
                raise ValueError(f"{name} conflicts with result physical parameters")
            setattr(self.metadata, name, value)

        if self.is_time_varying_power:
            if self.metadata.characteristic_power is not None:
                raise ValueError(
                    "metadata.characteristic_power is scalar and cannot represent time-varying P_c"
                )
            self.metadata.extra["characteristic_power_mode"] = "time_varying"
        else:
            power = float(self.P_c)
            current = self.metadata.characteristic_power
            if current is not None and current != power:
                raise ValueError("characteristic_power conflicts with result P_c")
            self.metadata.characteristic_power = power

    def __len__(self) -> int:
        return int(self.xi.size)

    @property
    def is_time_varying_power(self) -> bool:
        return np.asarray(self.P_c).ndim == 1

    @property
    def b_abs(self) -> np.ndarray:
        return np.abs(self.b)

    @property
    def beta_abs(self) -> np.ndarray:
        return np.abs(self.beta)

    @property
    def U_abs(self) -> np.ndarray:
        return np.abs(self.U)

    @property
    def eta(self) -> np.ndarray:
        """Return ``|b| / P_c`` where ``P_c > 0`` and ``NaN`` where ``P_c = 0``."""
        power = np.asarray(self.P_c, dtype=float)
        if power.ndim == 0:
            power = np.full(len(self), float(power), dtype=float)
        out = np.full(len(self), np.nan, dtype=float)
        np.divide(self.b_abs, power, out=out, where=power > 0.0)
        return out

    @property
    def b_mean(self) -> float:
        return float(np.mean(self.b_abs))

    @property
    def beta_mean(self) -> float:
        return float(np.mean(self.beta_abs))

    @property
    def theta_mean(self) -> float:
        return float(np.mean(self.theta))

    @property
    def theta_std(self) -> float:
        return float(np.std(self.theta))

    @property
    def A_ref_unit(self) -> str:
        return self.unit

    @property
    def tau_unit(self) -> str:
        return self.coordinate_unit

    @property
    def b_unit(self) -> str:
        return self.metadata.agencity_unit

    @property
    def memory_window(self) -> float | None:
        return self.metadata.memory_window

    def summary(self) -> dict[str, Any]:
        """Return concise descriptive statistics without attaching diagnostics."""
        power = np.asarray(self.P_c, dtype=float)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "n_samples": len(self),
            "tau": self.tau,
            "P_c": None if self.is_time_varying_power else float(self.P_c),
            "P_c_mean": float(np.mean(power)),
            "P_c_time_varying": self.is_time_varying_power,
            "A_ref": self.A_ref,
            "memory_window": self.memory_window,
            "b_mean": self.b_mean,
            "b_std": float(np.std(self.b_abs)),
            "b_peak": float(np.max(self.b_abs)),
            "beta_mean": self.beta_mean,
            "beta_max": float(np.max(self.beta_abs)),
            "J_mean": float(np.mean(self.J)),
            "D_mean": float(np.mean(self.D)),
            "S_mean": float(np.mean(self.S)),
            "M_mean": float(np.mean(self.M)),
            "O_mean": float(np.mean(self.O)),
            "theta_mean": self.theta_mean,
            "theta_std": self.theta_std,
            "unit": self.unit,
            "coordinate_unit": self.coordinate_unit,
            "power_unit": self.power_unit,
            "b_unit": self.b_unit,
            "observable_kind": self.observable_kind,
            "domain": self.domain,
            "system_type": self.system_type,
            "mechanism": self.mechanism,
        }

    def to_dict(self) -> dict[str, Any]:
        from agencitylab.io.result_serialization import result_to_dict

        return result_to_dict(self, schema_version=RESULT_SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgencityResult:
        from agencitylab.io.result_serialization import result_from_dict

        return result_from_dict(cls, data, schema_version=RESULT_SCHEMA_VERSION)

    def to_dataframe(self):
        from agencitylab.io.adapters import result_to_dataframe

        return result_to_dataframe(self)

    def to_xarray(self):
        from agencitylab.io.adapters import result_to_xarray

        return result_to_xarray(self, schema_version=RESULT_SCHEMA_VERSION)

    def save_json(self, path: str | Path) -> Path:
        from agencitylab.io.save import save

        return save(self.to_dict(), path)

    @classmethod
    def load_json(cls, path: str | Path) -> AgencityResult:
        from agencitylab.io.load import load

        return cls.from_dict(load(path))
