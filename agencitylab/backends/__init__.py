"""
Backend selection for AgencityLab.

The framework ships with a NumPy backend by default and can optionally
use Numba or JAX when they are installed.
"""

from .numpy_backend import (
    apply_window_numpy,
    central_difference_numpy,
    causal_moving_correlation_numpy,
    normalize_numpy,
)
from .selector import (
    BackendUnavailableError,
    get_backend,
    get_backend_name,
    has_jax,
    has_numba,
    select_backend,
)

__all__ = [
    "BackendUnavailableError",
    "apply_window_numpy",
    "central_difference_numpy",
    "causal_moving_correlation_numpy",
    "get_backend",
    "get_backend_name",
    "has_jax",
    "has_numba",
    "normalize_numpy",
    "select_backend",
]
