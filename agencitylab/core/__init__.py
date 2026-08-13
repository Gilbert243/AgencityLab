"""Canonical mathematical engine for AgencityLab.

The core namespace contains deterministic primitives for the accepted scalar
pipeline:

``u -> u* -> X* -> A* -> M,O -> D,S -> J,U,Theta -> beta -> b``.

Interpretation, coherence diagnostics, regimes and reports belong to
:mod:`agencitylab.analysis`. High-level orchestration belongs to
:func:`agencitylab.compute_agencity`.
"""

from .activation import activation, reduced_coordinate
from .activity import activity
from .agencity import agencity
from .beta import beta, compute_beta, structured_agencity
from .contrast import compute_contrast
from .crm import causal_moving_correlation, crm_tau
from .intensity import (
    compute_dynamic_intensity,
    compute_intensities,
    compute_structural_intensity,
)
from .memory import memory
from .normalization import normalize_signal
from .organization import organization
from .orientation import compute_angle, compute_orientation
from .power import characteristic_power
from .tau import characteristic_time
from .validation import (
    as_float_array,
    validate_axis,
    validate_signal,
    validate_window_size,
)

__all__ = [
    "normalize_signal",
    "activation",
    "reduced_coordinate",
    "activity",
    "causal_moving_correlation",
    "crm_tau",
    "memory",
    "organization",
    "compute_dynamic_intensity",
    "compute_structural_intensity",
    "compute_intensities",
    "compute_contrast",
    "compute_orientation",
    "compute_angle",
    "beta",
    "structured_agencity",
    "compute_beta",
    "agencity",
    "characteristic_power",
    "characteristic_time",
    "as_float_array",
    "validate_axis",
    "validate_signal",
    "validate_window_size",
]
