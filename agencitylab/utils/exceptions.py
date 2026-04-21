"""
Custom exception hierarchy for AgencityLab.
"""

from __future__ import annotations


class AgencityError(Exception):
    """Base class for all AgencityLab exceptions."""


class DataValidationError(AgencityError):
    """Raised when raw inputs fail validation."""


class OptionalDependencyError(AgencityError):
    """Raised when an optional dependency is required but missing."""
