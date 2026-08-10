"""Backend selection and optional acceleration primitives for AgencityLab."""

from .jax_backend import (
    apply_window_jax,
    causal_moving_correlation_jax,
    central_difference_jax,
    normalize_jax,
)
from .numba_backend import (
    apply_window_numba,
    causal_moving_correlation_numba,
    central_difference_numba,
    normalize_numba,
)
from .numpy_backend import (
    apply_window_numpy,
    causal_moving_correlation_numpy,
    central_difference_numpy,
    normalize_numpy,
)
from .selector import (
    BackendUnavailableError,
    available_backends,
    backend_capabilities,
    get_backend,
    get_backend_name,
    has_jax,
    has_numba,
    select_backend,
)

__all__ = [
    "BackendUnavailableError",
    "available_backends",
    "backend_capabilities",
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
