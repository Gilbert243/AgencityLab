"""
Validation helpers for AgencityLab configuration objects.

This module performs lightweight validation without introducing heavy
dependencies such as Pydantic by default.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from .defaults import AgencityConfig
from .modes import AgencityMode


def validate_config(config: Any) -> AgencityConfig:
    """
    Validate and normalize a configuration object.

    Parameters
    ----------
    config:
        Either an AgencityConfig instance or a mapping compatible with it.

    Returns
    -------
    AgencityConfig
        Normalized validated configuration object.
    """
    if isinstance(config, AgencityConfig):
        _validate_config_values(config)
        return config

    if isinstance(config, Mapping):
        normalized = AgencityConfig.from_dict(dict(config))
        _validate_config_values(normalized)
        return normalized

    if is_dataclass(config):
        normalized = AgencityConfig.from_dict(config.__dict__)
        _validate_config_values(normalized)
        return normalized

    raise TypeError("config must be an AgencityConfig or a mapping.")


def _validate_config_values(config: AgencityConfig) -> None:
    """Check all numeric and categorical configuration fields."""
    if not isinstance(config.mode, AgencityMode):
        raise TypeError("mode must be an AgencityMode instance.")

    if not (0.0 < float(config.tau_threshold) < 1.0):
        raise ValueError("tau_threshold must lie strictly between 0 and 1.")

    if int(config.activity_window) < 1:
        raise ValueError("activity_window must be >= 1.")

    if int(config.crm_window) < 1:
        raise ValueError("crm_window must be >= 1.")

    if float(config.epsilon) <= 0.0:
        raise ValueError("epsilon must be strictly positive.")
