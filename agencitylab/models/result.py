"""Stable result model returned by the AgencityLab computational API."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional

import numpy as np

from .metadata import ExperimentMetadata

RESULT_SCHEMA_VERSION = "0.3"


def _serialize(value: Any) -> Any:
    """Recursively convert NumPy, complex, and dataclass values to JSON-safe data."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize(value.to_dict())
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "__complex_array__": True,
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        value = complex(value)
        return {"__complex__": True, "real": value.real, "imag": value.imag}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    return value


def _deserialize(value: Any) -> Any:
    """Restore structures emitted by :func:`_serialize`."""
    if isinstance(value, dict):
        if value.get("__complex__"):
            return complex(value["real"], value["imag"])
        if value.get("__complex_array__"):
            real = np.asarray(value["real"], dtype=float)
            imag = np.asarray(value["imag"], dtype=float)
            return real + 1j * imag
        return {key: _deserialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize(item) for item in value]
    return value


def _positive_scalar(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a scalar number") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite")
    return out


def _nonnegative_power(value: Any, *, n: int):
    """Validate a finite scalar or sampled characteristic-power profile ``P_c >= 0``."""
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


def _finite_1d(value: Any, *, name: str, dtype=None) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


@dataclass(slots=True)
class AgencityResult:
    """Canonical scalar-signal result container with a stable serialization contract."""

    xi: np.ndarray
    u: np.ndarray
    u_star: np.ndarray
    X_star: np.ndarray
    A_star: np.ndarray
    t_star: np.ndarray
    tau: float
    P_c: Any

    A_ref: float = 1.0

    # Compatibility fields retained from pre-v0.3 objects. They are not canonical
    # modifiers of the current computation path.
    A_fact: float = 1.0
    resolution_scale: Optional[float] = None

    M: np.ndarray = field(default_factory=lambda: np.array([]))
    O: np.ndarray = field(default_factory=lambda: np.array([]))
    D: np.ndarray = field(default_factory=lambda: np.array([]))
    S: np.ndarray = field(default_factory=lambda: np.array([]))
    J: np.ndarray = field(default_factory=lambda: np.array([]))
    U: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    beta: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    b_reduced: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    b: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    theta: Optional[np.ndarray] = None

    unit: str = ""
    coordinate_unit: str = ""
    power_unit: str = ""
    observable_kind: str = ""
    domain: str = ""
    system_type: str = ""
    mechanism: str = ""

    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    config: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[dict] = None
    multiscale: Optional[Any] = None
    report_text: Optional[str] = None

    def __post_init__(self) -> None:
        self.xi = _finite_1d(self.xi, name="xi", dtype=float)
        if self.xi.size < 3:
            raise ValueError("xi must contain at least three samples")
        if np.any(np.diff(self.xi) <= 0.0):
            raise ValueError("xi must be strictly increasing")

        n = self.xi.size
        real_arrays = (
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
        )
        for name in real_arrays:
            arr = _finite_1d(getattr(self, name), name=name, dtype=float)
            if arr.size != n:
                raise ValueError(f"{name} length must match xi")
            setattr(self, name, arr)

        for name in ("U", "beta", "b_reduced", "b"):
            arr = _finite_1d(getattr(self, name), name=name, dtype=complex)
            if arr.size != n:
                raise ValueError(f"{name} length must match xi")
            setattr(self, name, arr)

        self.tau = _positive_scalar(self.tau, name="tau")
        self.P_c = _nonnegative_power(self.P_c, n=n)
        self.A_ref = _positive_scalar(self.A_ref, name="A_ref")
        self.A_fact = _positive_scalar(self.A_fact, name="A_fact")
        if self.resolution_scale is not None:
            self.resolution_scale = _positive_scalar(
                self.resolution_scale, name="resolution_scale"
            )

        self.metadata = ExperimentMetadata.from_dict(self.metadata)
        self.config = dict(self.config or {})
        self.analysis = dict(self.analysis or {})

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
            # Canonical Theta is atan2(O, M), represented by angle(U). Do not
            # unwrap here: phase unwrapping belongs to analysis, not the result model.
            self.theta = np.angle(self.U)
        else:
            self.theta = _finite_1d(self.theta, name="theta", dtype=float)
            if self.theta.size != n:
                raise ValueError("theta length must match xi")

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
                raise ValueError(
                    f"{result_name} conflicts with metadata.{metadata_name}"
                )
            if not result_value and metadata_value:
                setattr(self, result_name, metadata_value)
            elif result_value and not metadata_value:
                setattr(self.metadata, metadata_name, result_value)

        for name, value in (
            ("reference_amplitude", self.A_ref),
            ("characteristic_time", self.tau),
        ):
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
        """Whether ``P_c`` is represented by one value per sample."""
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
        """Return ``|b| / P_c`` where ``P_c > 0`` and ``NaN`` where ``P_c = 0``.

        The canonical computation itself already stores ``beta``. At zero power,
        the inverse ratio ``|b| / P_c`` is undefined and is therefore not repaired
        with epsilon or silently replaced by ``|beta|``.
        """
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
    def memory_window(self) -> Optional[float]:
        return self.metadata.memory_window

    def summary(self) -> Dict[str, Any]:
        power = np.asarray(self.P_c, dtype=float)
        power_mean = float(np.mean(power))
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "n_samples": len(self),
            "tau": self.tau,
            "P_c": None if self.is_time_varying_power else float(self.P_c),
            "Pc_mean": power_mean,
            "P_c_time_varying": self.is_time_varying_power,
            "A_ref": self.A_ref,
            "A_fact": self.A_fact,
            "resolution_scale": self.resolution_scale,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "xi": _serialize(self.xi),
            "u": _serialize(self.u),
            "u_star": _serialize(self.u_star),
            "X_star": _serialize(self.X_star),
            "A_star": _serialize(self.A_star),
            "t_star": _serialize(self.t_star),
            "tau": self.tau,
            "P_c": _serialize(self.P_c),
            "A_ref": self.A_ref,
            "A_fact": self.A_fact,
            "resolution_scale": self.resolution_scale,
            "M": _serialize(self.M),
            "O": _serialize(self.O),
            "D": _serialize(self.D),
            "S": _serialize(self.S),
            "J": _serialize(self.J),
            "U": _serialize(self.U),
            "beta": _serialize(self.beta),
            "b_reduced": _serialize(self.b_reduced),
            "b": _serialize(self.b),
            "theta": _serialize(self.theta),
            "unit": self.unit,
            "coordinate_unit": self.coordinate_unit,
            "power_unit": self.power_unit,
            "observable_kind": self.observable_kind,
            "domain": self.domain,
            "system_type": self.system_type,
            "mechanism": self.mechanism,
            "metadata": self.metadata.to_dict(),
            "config": _serialize(self.config),
            "analysis": _serialize(self.analysis),
            "signature": _serialize(self.signature),
            "multiscale": _serialize(self.multiscale),
            "report_text": self.report_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencityResult":
        if not isinstance(data, dict):
            raise ValueError("result payload must be a dictionary")
        payload = dict(data)
        if "xi" not in payload:
            raise ValueError("xi is required")

        xi = np.asarray(_deserialize(payload["xi"]), dtype=float)
        n = len(xi)
        metadata = ExperimentMetadata.from_dict(payload.get("metadata"))

        def array(key: str, *, value=0.0, dtype=float):
            if key not in payload:
                return np.full(n, value, dtype=dtype)
            return np.asarray(_deserialize(payload[key]), dtype=dtype)

        def physical(
            key: str,
            aliases: tuple[str, ...],
            default: float,
            *,
            metadata_value=None,
        ) -> Any:
            if key in payload:
                return _deserialize(payload[key])
            for alias in aliases:
                if alias in payload:
                    warnings.warn(
                        f"result field '{alias}' is deprecated; use '{key}'",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return _deserialize(payload[alias])
            if metadata_value is not None:
                return metadata_value
            warnings.warn(
                f"legacy result payload is missing '{key}'; default {default} was applied",
                DeprecationWarning,
                stacklevel=2,
            )
            return default

        beta_value = array("beta", dtype=complex)

        return cls(
            xi=xi,
            u=array("u"),
            u_star=array("u_star"),
            X_star=array("X_star"),
            A_star=array("A_star"),
            t_star=array("t_star"),
            tau=physical(
                "tau",
                ("characteristic_time",),
                1.0,
                metadata_value=metadata.characteristic_time,
            ),
            P_c=physical(
                "P_c",
                ("Pc", "characteristic_power"),
                1.0,
                metadata_value=metadata.characteristic_power,
            ),
            A_ref=physical(
                "A_ref",
                ("reference_amplitude",),
                1.0,
                metadata_value=metadata.reference_amplitude,
            ),
            A_fact=payload.get("A_fact", 1.0),
            resolution_scale=payload.get("resolution_scale"),
            M=array("M"),
            O=array("O"),
            D=array("D"),
            S=array("S"),
            J=array("J"),
            U=array("U", dtype=complex),
            beta=beta_value,
            b_reduced=(
                array("b_reduced", dtype=complex)
                if "b_reduced" in payload
                else beta_value.copy()
            ),
            b=array("b", dtype=complex),
            theta=(array("theta") if "theta" in payload else None),
            unit=payload.get("unit", ""),
            coordinate_unit=payload.get("coordinate_unit", ""),
            power_unit=payload.get("power_unit", ""),
            observable_kind=payload.get("observable_kind", ""),
            domain=payload.get("domain", ""),
            system_type=payload.get("system_type", ""),
            mechanism=payload.get("mechanism", ""),
            metadata=metadata,
            config=dict(_deserialize(payload.get("config", {}))),
            analysis=dict(_deserialize(payload.get("analysis", {}))),
            signature=_deserialize(payload.get("signature")),
            multiscale=_deserialize(payload.get("multiscale")),
            report_text=payload.get("report_text"),
        )

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pandas required") from exc

        power = (
            self.P_c
            if self.is_time_varying_power
            else np.full(len(self), float(self.P_c), dtype=float)
        )
        frame = pd.DataFrame(
            {
                "xi": self.xi,
                "u": self.u,
                "u_star": self.u_star,
                "X_star": self.X_star,
                "A_star": self.A_star,
                "M": self.M,
                "O": self.O,
                "D": self.D,
                "S": self.S,
                "J": self.J,
                "theta": self.theta,
                "P_c": power,
                "beta_real": self.beta.real,
                "beta_imag": self.beta.imag,
                "beta_abs": self.beta_abs,
                "b_real": self.b.real,
                "b_imag": self.b.imag,
                "b_abs": self.b_abs,
            }
        )
        frame.attrs["units"] = self.metadata.unit_contract()
        return frame

    def to_xarray(self):
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("xarray required") from exc

        data_vars = {
            "u": ("xi", self.u),
            "u_star": ("xi", self.u_star),
            "X_star": ("xi", self.X_star),
            "A_star": ("xi", self.A_star),
            "M": ("xi", self.M),
            "O": ("xi", self.O),
            "D": ("xi", self.D),
            "S": ("xi", self.S),
            "J": ("xi", self.J),
            "theta": ("xi", self.theta),
            "beta_abs": ("xi", self.beta_abs),
            "b_abs": ("xi", self.b_abs),
        }
        attrs = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "tau": self.tau,
            "A_ref": self.A_ref,
            "unit": self.unit,
            "coordinate_unit": self.coordinate_unit,
            "power_unit": self.power_unit,
            "b_unit": self.b_unit,
        }
        if self.is_time_varying_power:
            data_vars["P_c"] = ("xi", self.P_c)
        else:
            attrs["P_c"] = float(self.P_c)

        dataset = xr.Dataset(data_vars=data_vars, coords={"xi": self.xi}, attrs=attrs)
        dataset["xi"].attrs["unit"] = self.coordinate_unit
        dataset["u"].attrs["unit"] = self.unit
        dataset["b_abs"].attrs["unit"] = self.b_unit
        if "P_c" in dataset:
            dataset["P_c"].attrs["unit"] = self.power_unit
        return dataset

    def save_json(self, path):
        from agencitylab.io.save import save

        return save(self.to_dict(), path)

    @classmethod
    def load_json(cls, path):
        from agencitylab.io.load import load

        return cls.from_dict(load(path))

    def attach_analysis(self, analysis):
        self.analysis = dict(analysis or {})
        return self

    def attach_report(self, report_text):
        self.report_text = report_text
        return self
