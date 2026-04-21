"""
Data preparation layer for AgencityLab.

This package transforms heterogeneous inputs into canonical signals u(ξ)
that can be consumed by the core Agencity computations.
"""

from .pipeline import DataPipeline, prepare_signal
from .validation import validate_input, validate_signal_input

__all__ = [
    "DataPipeline",
    "prepare_signal",
    "validate_input",
    "validate_signal_input",
]
