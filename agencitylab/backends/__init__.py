"""
Backend selection and acceleration layer for AgencityLab.
"""

from .selector import (
    BackendUnavailableError,
    available_backends,
    get_backend,
    get_backend_name,
    has_jax,
    has_numba,
    select_backend,
)

from .numpy_backend import (
    normalize_numpy,
    central_difference_numpy,
    apply_window_numpy,
    causal_moving_correlation_numpy,
)

from .numba_backend import (
    normalize_numba,
    central_difference_numba,
    apply_window_numba,
    causal_moving_correlation_numba,
)

from .jax_backend import (
    normalize_jax,
    central_difference_jax,
    apply_window_jax,
    causal_moving_correlation_jax,
)

__all__ = [
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
    "get_backend_name",
    "has_jax",
    "has_numba",
    "select_backend",
    "normalize_numpy",
    "central_difference_numpy",
    "apply_window_numpy",
    "causal_moving_correlation_numpy",
    "normalize_numba",
    "central_difference_numba",
    "apply_window_numba",
    "causal_moving_correlation_numba",
    "normalize_jax",
    "central_difference_jax",
    "apply_window_jax",
    "causal_moving_correlation_jax",
]