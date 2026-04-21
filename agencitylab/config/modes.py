"""
Operational modes for AgencityLab.

The mode controls the level of strictness, optional accelerators and
experimental features that are allowed at runtime.
"""

from __future__ import annotations

from enum import Enum


class AgencityMode(str, Enum):
    """Canonical operational modes for the framework."""

    CANONICAL = "canonical"
    EXPERIMENTAL = "experimental"
    FAST = "fast"
    DEBUG = "debug"

    @classmethod
    def from_value(cls, value: object) -> "AgencityMode":
        """Parse a mode from a string or an existing enum value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(
            "Unknown Agencity mode. Expected one of: canonical, experimental, fast, debug."
        )
