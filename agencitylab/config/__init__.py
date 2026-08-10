"""
Configuration layer for AgencityLab.

This package provides:
- operational modes
- validated configuration objects
- runtime registry
"""

from .modes import AgencityMode
from .defaults import AgencityConfig, DEFAULT_CONFIG
from .schema import validate_config
from .runtime import (
    get_runtime_config,
    set_runtime_config,
    update_runtime_config,
    reset_runtime_config,
    use_config,
)

__all__ = [
    "AgencityMode",
    "AgencityConfig",
    "DEFAULT_CONFIG",
    "validate_config",
    "get_runtime_config",
    "set_runtime_config",
    "update_runtime_config",
    "reset_runtime_config",
    "use_config",
]