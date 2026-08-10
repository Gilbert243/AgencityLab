"""Public exception hierarchy for AgencityLab.

The public API uses typed exceptions so applications can distinguish invalid
inputs, missing physical context, unit-label problems, batch failures, and
streaming state errors without parsing arbitrary error strings.
"""

from __future__ import annotations


class AgencityError(Exception):
    """Base class for public AgencityLab errors."""


class AgencityValidationError(AgencityError, ValueError):
    """Invalid user input or unsupported public-API option."""


class PhysicalParameterError(AgencityValidationError):
    """Missing or invalid physical/contextual parameter."""


class UnitValidationError(AgencityValidationError):
    """Invalid unit metadata or unit-label contract."""


class BatchItemError(AgencityError):
    """Failure while computing one item in a batch."""


class StreamStateError(AgencityError, RuntimeError):
    """Invalid streaming state or chunk ordering."""


class StreamNotReadyError(StreamStateError):
    """The stream buffer does not yet contain enough data to compute."""


__all__ = [
    "AgencityError",
    "AgencityValidationError",
    "PhysicalParameterError",
    "UnitValidationError",
    "BatchItemError",
    "StreamStateError",
    "StreamNotReadyError",
]
