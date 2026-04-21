"""
Runtime configuration registry.

This module stores the active configuration used by the current process.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from .defaults import AgencityConfig, DEFAULT_CONFIG
from .schema import validate_config

_RUNTIME_CONFIG: AgencityConfig = DEFAULT_CONFIG


def get_runtime_config() -> AgencityConfig:
    """Return the active runtime configuration."""
    return _RUNTIME_CONFIG


def set_runtime_config(config: AgencityConfig) -> AgencityConfig:
    """
    Replace the active runtime configuration.

    Returns the normalized configuration object.
    """
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = validate_config(config)
    return _RUNTIME_CONFIG


def reset_runtime_config() -> AgencityConfig:
    """Reset the runtime configuration to the default canonical config."""
    return set_runtime_config(DEFAULT_CONFIG)


@contextmanager
def use_config(config: AgencityConfig) -> Iterator[AgencityConfig]:
    """
    Temporarily use a different configuration inside a context manager.
    """
    global _RUNTIME_CONFIG
    previous = _RUNTIME_CONFIG
    try:
        yield set_runtime_config(config)
    finally:
        _RUNTIME_CONFIG = previous
