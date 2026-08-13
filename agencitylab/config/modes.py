"""Operational software modes.

Modes are software-policy labels for strictness, optional acceleration, and
feature exposure. They do not select alternate scientific equations and do not
supply physical parameters.
"""

from __future__ import annotations

from enum import Enum


class AgencityMode(str, Enum):
    """Software execution mode; never a replacement for scientific status."""

    CANONICAL = "canonical"
    EXPERIMENTAL = "experimental"
    FAST = "fast"
    DEBUG = "debug"

    @classmethod
    def from_value(cls, value: object) -> "AgencityMode":
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
