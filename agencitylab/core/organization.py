"""Organization operator O.

Canonical formula:
    O(t*) = tanh(crm_tau(X*)(t*))
"""

from __future__ import annotations

import numpy as np

from .crm import causal_moving_correlation
from .safeguards import safe_tanh


def organization(activation_signal, window_size=1.0, *, axis=None, return_correlation: bool = False):
    """Compute the organization component from the activation signal."""
    crm = causal_moving_correlation(activation_signal, window_size=window_size, axis=axis)
    org = safe_tanh(crm)
    if return_correlation:
        return org, crm
    return org


def organization_from_signal(*args, **kwargs):
    """Alias kept for API clarity."""
    return organization(*args, **kwargs)
