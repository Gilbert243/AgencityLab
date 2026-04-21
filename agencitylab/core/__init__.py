"""Core mathematical engine for AgencityLab.

The core layer contains the canonical formulas and the numerically stable
helpers required to compute:
    u -> u* -> X* -> A* -> M -> O -> beta -> b
"""

from .normalization import center_signal, compute_reference_scale, normalize_signal
from .activation import activation, activation_from_signal, reduced_coordinate
from .activity import activity, activity_from_signal
from .autocorr import autocorrelation, normalized_autocorrelation
from .tau import estimate_tau
from .crm import causal_moving_correlation, crm_tau
from .memory import memory, memory_from_signal
from .organization import organization, organization_from_signal
from .beta import beta, structured_agency
from .agencity import agencity, agencity_rate
from .power import characteristic_power, estimate_characteristic_power
from .safeguards import EPS, safe_divide, safe_tanh, saturate
from .validation import (
    as_float_array,
    validate_axis,
    validate_signal,
    validate_window_size,
)

__all__ = [
    "center_signal",
    "compute_reference_scale",
    "normalize_signal",
    "activation",
    "activation_from_signal",
    "reduced_coordinate",
    "activity",
    "activity_from_signal",
    "autocorrelation",
    "normalized_autocorrelation",
    "estimate_tau",
    "causal_moving_correlation",
    "crm_tau",
    "memory",
    "memory_from_signal",
    "organization",
    "organization_from_signal",
    "beta",
    "structured_agency",
    "agencity",
    "agencity_rate",
    "characteristic_power",
    "estimate_characteristic_power",
    "EPS",
    "safe_divide",
    "safe_tanh",
    "saturate",
    "as_float_array",
    "validate_axis",
    "validate_signal",
    "validate_window_size",
]
