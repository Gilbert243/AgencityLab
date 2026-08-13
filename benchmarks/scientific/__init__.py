"""Deterministic scientific reference benchmarks for AgencityLab."""

from .reference_bench import (
    ReferenceSignal,
    circular_variance,
    periodic_relative_error,
    reference_suite,
    structural_mask,
)

__all__ = [
    "ReferenceSignal",
    "circular_variance",
    "periodic_relative_error",
    "reference_suite",
    "structural_mask",
]
