"""Validation for software/runtime configuration only."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from .defaults import AgencityConfig
from .modes import AgencityMode


def validate_config(config: Any) -> AgencityConfig:
    """Normalize a config-like object into a validated software configuration."""

    if isinstance(config, AgencityConfig):
        _validate(config)
        return config
    if isinstance(config, Mapping):
        cfg = AgencityConfig.from_dict(dict(config))
        _validate(cfg)
        return cfg
    if is_dataclass(config):
        payload = {
            field_name: getattr(config, field_name)
            for field_name in getattr(config, "__dataclass_fields__", {})
        }
        cfg = AgencityConfig.from_dict(payload)
        _validate(cfg)
        return cfg
    raise TypeError("config must be AgencityConfig or mapping.")


def _validate(c: AgencityConfig) -> None:
    if not isinstance(c.mode, AgencityMode):
        raise TypeError("mode invalid")

    normalization = str(c.normalization_method).strip().lower()
    if normalization not in {"a_ref", "canonical", "auto", "default"}:
        raise ValueError("normalization_method must preserve canonical A_ref normalization")

    backend = str(c.backend).lower().strip()
    if backend not in {"numpy", "numba", "jax", "auto"}:
        raise ValueError("backend must be one of: numpy, numba, jax, auto")
    if c.report_language not in {"en", "fr"}:
        raise ValueError("report_language must be one of: en, fr")
    if not isinstance(c.metadata, dict):
        raise TypeError("metadata must be a dictionary")
