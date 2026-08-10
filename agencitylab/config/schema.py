from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from .defaults import AgencityConfig
from .modes import AgencityMode


def validate_config(config: Any) -> AgencityConfig:
    """
    Normalize any config-like object into a valid AgencityConfig.
    """
    if isinstance(config, AgencityConfig):
        _validate(config)
        return config

    if isinstance(config, Mapping):
        cfg = AgencityConfig.from_dict(dict(config))
        _validate(cfg)
        return cfg

    if is_dataclass(config):
        cfg = AgencityConfig.from_dict(config.__dict__)
        _validate(cfg)
        return cfg

    raise TypeError("config must be AgencityConfig or mapping.")


def _validate(c: AgencityConfig) -> None:
    """
    Validate the configuration values.
    """
    if not isinstance(c.mode, AgencityMode):
        raise TypeError("mode invalid")

    if not (0 < c.tau_threshold < 1):
        raise ValueError("tau_threshold ∈ (0,1)")

    if c.activity_window < 1:
        raise ValueError("activity_window ≥ 1")

    if c.crm_window < 1:
        raise ValueError("crm_window ≥ 1")

    if c.epsilon <= 0:
        raise ValueError("epsilon > 0")

    if c.reduced_time_step <= 0:
        raise ValueError("reduced_time_step > 0")

    if c.agencity_scale <= 0:
        raise ValueError("agencity_scale > 0")

    if c.temperature <= 0:
        raise ValueError("temperature > 0")

    backend = str(c.backend).lower().strip()
    if backend not in {"numpy", "numba", "jax", "auto"}:
        raise ValueError("backend must be one of: numpy, numba, jax, auto")

    if c.metric_type not in {"identity", "diagonal", "learned"}:
        raise ValueError("metric_type must be one of: identity, diagonal, learned")

    if c.report_language not in {"en", "fr"}:
        raise ValueError("report_language must be one of: en, fr")