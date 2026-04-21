"""Memory operator M.

Canonical formula:
    M(t*) = tanh(crm_tau(A*)(t*))
"""

from __future__ import annotations

import numpy as np

from .crm import causal_moving_correlation
from .safeguards import safe_tanh


def memory(activity_signal, window_size=1.0, *, axis=None, return_correlation: bool = False):
    """Compute the memory component from the activity signal."""
    crm = causal_moving_correlation(activity_signal, window_size=window_size, axis=axis)
    crm = np.nan_to_num(crm, nan=0.0)
    
    #mem = safe_tanh(crm)
    
    # 🔥 compression douce (très important)
    mem = np.tanh(0.5 * crm)
    if return_correlation:
        return mem, crm
    return mem


def memory_from_signal(*args, **kwargs):
    """Alias kept for API clarity."""
    return memory(*args, **kwargs)
