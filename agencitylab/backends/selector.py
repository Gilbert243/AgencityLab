"""
Backend selection utilities.

The selector provides a single entry point for choosing the computational
backend used by optional layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .numpy_backend import (
    apply_window_numpy,
    central_difference_numpy,
    causal_moving_correlation_numpy,
    normalize_numpy,
)

BackendName = Literal["numpy", "numba", "jax"]


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend is not available."""


def has_numba() -> bool:
    """Return True if Numba is installed."""
    try:
        from .numba_backend import has_numba as _has_numba
        return _has_numba()
    except Exception:
        return False


def has_jax() -> bool:
    """Return True if JAX is installed."""
    try:
        from .jax_backend import has_jax as _has_jax
        return _has_jax()
    except Exception:
        return False


def get_backend_name(preferred: Optional[str] = None) -> BackendName:
    """
    Resolve the backend name from a preferred choice.
    """
    if preferred is None:
        return "numpy"

    preferred = preferred.lower().strip()

    if preferred == "jax":
        if has_jax():
            return "jax"
        raise BackendUnavailableError("Requested JAX backend is unavailable.")

    if preferred == "numba":
        if has_numba():
            return "numba"
        raise BackendUnavailableError("Requested Numba backend is unavailable.")

    if preferred == "numpy":
        return "numpy"

    raise ValueError("Unknown backend name.")


def select_backend(preferred: Optional[str] = None):
    """
    Return a backend module-like object.

    The result is intentionally lightweight and only guarantees the methods
    used by the base framework.
    """
    name = get_backend_name(preferred)

    if name == "numpy":
        return {
            "normalize": normalize_numpy,
            "central_difference": central_difference_numpy,
            "apply_window": apply_window_numpy,
            "causal_moving_correlation": causal_moving_correlation_numpy,
        }

    if name == "numba":
        from .numba_backend import normalize_numba
        return {
            "normalize": normalize_numba,
            "central_difference": central_difference_numpy,
            "apply_window": apply_window_numpy,
            "causal_moving_correlation": causal_moving_correlation_numpy,
        }

    if name == "jax":
        from .jax_backend import normalize_jax
        return {
            "normalize": normalize_jax,
            "central_difference": central_difference_numpy,
            "apply_window": apply_window_numpy,
            "causal_moving_correlation": causal_moving_correlation_numpy,
        }

    raise BackendUnavailableError(f"Unsupported backend: {name}")


def get_backend(preferred: Optional[str] = None):
    """Alias for select_backend()."""
    return select_backend(preferred)
