"""Serialization contract for :class:`agencitylab.models.AgencityResult`."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, cast

import numpy as np

from agencitylab.models.metadata import ExperimentMetadata


def serialize_value(value: Any) -> Any:
    """Recursively convert NumPy, complex and dataclass values to JSON-safe data."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serialize_value(value.to_dict())
    if is_dataclass(value):
        return serialize_value(asdict(cast(Any, value)))
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "__complex_array__": True,
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        scalar = complex(value)
        return {"__complex__": True, "real": scalar.real, "imag": scalar.imag}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_value(item) for item in value]
    return value


def deserialize_value(value: Any) -> Any:
    """Restore structures emitted by :func:`serialize_value`."""
    if isinstance(value, dict):
        if value.get("__complex__"):
            return complex(value["real"], value["imag"])
        if value.get("__complex_array__"):
            real = np.asarray(value["real"], dtype=float)
            imag = np.asarray(value["imag"], dtype=float)
            return real + 1j * imag
        return {key: deserialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deserialize_value(item) for item in value]
    return value


def result_to_dict(result: Any, *, schema_version: str) -> dict[str, Any]:
    """Serialize a canonical result using the stable 1.0 result schema."""
    return {
        "schema_version": schema_version,
        "xi": serialize_value(result.xi),
        "u": serialize_value(result.u),
        "u_star": serialize_value(result.u_star),
        "X_star": serialize_value(result.X_star),
        "A_star": serialize_value(result.A_star),
        "t_star": serialize_value(result.t_star),
        "tau": result.tau,
        "P_c": serialize_value(result.P_c),
        "A_ref": result.A_ref,
        "M": serialize_value(result.M),
        "O": serialize_value(result.O),
        "D": serialize_value(result.D),
        "S": serialize_value(result.S),
        "J": serialize_value(result.J),
        "U": serialize_value(result.U),
        "beta": serialize_value(result.beta),
        "b": serialize_value(result.b),
        "theta": serialize_value(result.theta),
        "unit": result.unit,
        "coordinate_unit": result.coordinate_unit,
        "power_unit": result.power_unit,
        "observable_kind": result.observable_kind,
        "domain": result.domain,
        "system_type": result.system_type,
        "mechanism": result.mechanism,
        "metadata": result.metadata.to_dict(),
    }


def result_from_dict(result_cls: type, data: dict[str, Any], *, schema_version: str) -> Any:
    """Restore a result from the exact stable schema; no pre-1.0 migration is implicit."""
    if not isinstance(data, dict):
        raise ValueError("result payload must be a dictionary")
    payload = dict(data)
    received_version = payload.get("schema_version")
    if received_version != schema_version:
        raise ValueError(
            f"unsupported result schema {received_version!r}; expected {schema_version!r}"
        )

    required = {
        "xi",
        "u",
        "u_star",
        "X_star",
        "A_star",
        "t_star",
        "tau",
        "P_c",
        "A_ref",
        "M",
        "O",
        "D",
        "S",
        "J",
        "U",
        "beta",
        "b",
        "metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"result payload missing required field(s): {', '.join(missing)}")

    def array(key: str, *, dtype: Any = float) -> np.ndarray:
        return np.asarray(deserialize_value(payload[key]), dtype=dtype)

    metadata = ExperimentMetadata.from_dict(payload["metadata"])
    return result_cls(
        xi=array("xi"),
        u=array("u"),
        u_star=array("u_star"),
        X_star=array("X_star"),
        A_star=array("A_star"),
        t_star=array("t_star"),
        tau=float(payload["tau"]),
        P_c=deserialize_value(payload["P_c"]),
        A_ref=float(payload["A_ref"]),
        M=array("M"),
        O=array("O"),
        D=array("D"),
        S=array("S"),
        J=array("J"),
        U=array("U", dtype=complex),
        beta=array("beta", dtype=complex),
        b=array("b", dtype=complex),
        theta=array("theta") if payload.get("theta") is not None else None,
        unit=payload.get("unit", ""),
        coordinate_unit=payload.get("coordinate_unit", ""),
        power_unit=payload.get("power_unit", ""),
        observable_kind=payload.get("observable_kind", ""),
        domain=payload.get("domain", ""),
        system_type=payload.get("system_type", ""),
        mechanism=payload.get("mechanism", ""),
        metadata=metadata,
    )
