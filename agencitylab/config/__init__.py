"""Runtime software configuration for AgencityLab.

This namespace contains execution and UX options only. It is not a source of
physical parameters or scientific constants. Physical/contextual inputs belong
to the explicit scientific API that uses them.
"""

from .defaults import AgencityConfig, DEFAULT_CONFIG
from .modes import AgencityMode
from .runtime import (
    get_runtime_config,
    reset_runtime_config,
    set_runtime_config,
    update_runtime_config,
    use_config,
)
from .schema import validate_config

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
