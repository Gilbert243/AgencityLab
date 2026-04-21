"""Structured agencement beta.

Canonical formula:
    beta(t*) = tanh(X*(t*) * (1 + A*(t*))) * tanh(M(t*) + O(t*))
"""

from __future__ import annotations

import numpy as np

from .safeguards import safe_tanh


def beta(activation_signal, activity_signal, memory_signal, organization_signal):
    """Compute the structured agencement beta."""
    activation_signal = np.asarray(activation_signal, dtype=float)
    activity_signal = np.asarray(activity_signal, dtype=float)
    memory_signal = np.asarray(memory_signal, dtype=float)
    organization_signal = np.asarray(organization_signal, dtype=float)

    first = safe_tanh(activation_signal * (1.0 + activity_signal))
    second = safe_tanh(memory_signal + organization_signal)
    return first * second


def structured_agency(*args, **kwargs):
    """Alias for beta to keep the theory vocabulary explicit."""
    return beta(*args, **kwargs)
