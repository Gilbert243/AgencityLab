"""
Backend selection utilities.

The selector provides a single entry point for choosing the computational
backend used by optional layers.

Default behavior is conservative: NumPy is always safe and never blocks users.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from .numpy_backend import (
    apply_window_numpy,
    causal_moving_correlation_numpy,
    central_difference_numpy,
    normalize_numpy,
)

BackendName = Literal["numpy", "numba", "jax"]


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend is not available."""


@lru_cache(maxsize=None)
def has_numba() -> bool:
    """Return True if Numba is installed."""
    try:
        from .numba_backend import has_numba as _has_numba
        return _has_numba()
    except Exception:
        return False


@lru_cache(maxsize=None)
def has_jax() -> bool:
    """Return True if JAX is installed."""
    try:
        from .jax_backend import has_jax as _has_jax
        return _has_jax()
    except Exception:
        return False


def available_backends():
    """
    Return the list of available backends in order of preference.
    """
    backends = ["numpy"]
    if has_numba():
        backends.append("numba")
    if has_jax():
        backends.append("jax")
    return tuple(backends)


def get_backend_name(
    preferred: Optional[str] = None,
    *,
    auto: bool = False,
    prefer_gpu: bool = False,
) -> BackendName:
    """
    Resolve the backend name from a preferred choice.

    Rules
    -----
    - None => numpy (safe default)
    - auto/best => best available backend
    - explicit backend name => strict request, raises if unavailable
    """
    if preferred is None and not auto:
        return "numpy"

    if preferred is not None:
        preferred = preferred.lower().strip()

        if preferred in {"auto", "best"}:
            auto = True
        elif preferred in {"numpy", "numba", "jax"}:
            if preferred == "jax" and not has_jax():
                raise BackendUnavailableError("Requested JAX backend is unavailable.")
            if preferred == "numba" and not has_numba():
                raise BackendUnavailableError("Requested Numba backend is unavailable.")
            return preferred  # type: ignore[return-value]
        else:
            raise ValueError("Unknown backend name.")

    if auto:
        if prefer_gpu and has_jax():
            return "jax"
        if has_jax():
            return "jax"
        if has_numba():
            return "numba"
        return "numpy"

    return "numpy"


def select_backend(preferred: Optional[str] = None, *, auto: bool = False, prefer_gpu: bool = False):
    """
    Return a backend module-like dictionary.

    The result is intentionally lightweight and only guarantees the methods
    used by the base framework.
    """
    name = get_backend_name(preferred, auto=auto, prefer_gpu=prefer_gpu)

    if name == "numpy":
        return {
            "name": "numpy",
            "normalize": normalize_numpy,
            "central_difference": central_difference_numpy,
            "apply_window": apply_window_numpy,
            "causal_moving_correlation": causal_moving_correlation_numpy,
        }

    if name == "numba":
        from .numba_backend import (
            apply_window_numba,
            causal_moving_correlation_numba,
            central_difference_numba,
            normalize_numba,
        )
        return {
            "name": "numba",
            "normalize": normalize_numba,
            "central_difference": central_difference_numba,
            "apply_window": apply_window_numba,
            "causal_moving_correlation": causal_moving_correlation_numba,
        }

    if name == "jax":
        from .jax_backend import (
            apply_window_jax,
            causal_moving_correlation_jax,
            central_difference_jax,
            normalize_jax,
        )
        return {
            "name": "jax",
            "normalize": normalize_jax,
            "central_difference": central_difference_jax,
            "apply_window": apply_window_jax,
            "causal_moving_correlation": causal_moving_correlation_jax,
        }

    raise BackendUnavailableError(f"Unsupported backend: {name}")


def get_backend(preferred: Optional[str] = None, *, auto: bool = False, prefer_gpu: bool = False):
    """Alias for select_backend()."""
    return select_backend(preferred, auto=auto, prefer_gpu=prefer_gpu)