"""Backend selection and capability metadata for AgencityLab.

NumPy is the stable reference numerical engine for the canonical public
pipeline. Numba and JAX expose optional experimental primitives; selecting one
does not silently replace the canonical equations or their float64 reference
implementation.
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

_BACKEND_METADATA = {
    "numpy": {
        "status": "stable",
        "scope": "reference numerical primitives and canonical public pipeline",
        "canonical_pipeline": True,
        "default_dtype": "float64",
    },
    "numba": {
        "status": "experimental",
        "scope": "optional one-dimensional numerical primitives",
        "canonical_pipeline": False,
        "default_dtype": "float64",
    },
    "jax": {
        "status": "experimental",
        "scope": "optional autodiff/vectorisation primitives",
        "canonical_pipeline": False,
        "default_dtype": "float32 unless JAX x64 is explicitly enabled",
    },
}


class BackendUnavailableError(RuntimeError):
    """Raised when a requested optional backend is not available."""


@lru_cache(maxsize=None)
def has_numba() -> bool:
    """Return True if Numba is installed and importable."""
    try:
        from .numba_backend import has_numba as _has_numba
        return _has_numba()
    except Exception:
        return False


@lru_cache(maxsize=None)
def has_jax() -> bool:
    """Return True if JAX is installed and importable."""
    try:
        from .jax_backend import has_jax as _has_jax
        return _has_jax()
    except Exception:
        return False


def available_backends() -> tuple[BackendName, ...]:
    """Return installed backends, always beginning with the NumPy reference."""
    backends: list[BackendName] = ["numpy"]
    if has_numba():
        backends.append("numba")
    if has_jax():
        backends.append("jax")
    return tuple(backends)


def backend_capabilities(name: Optional[str] = None):
    """Return stability, scope, availability, and canonical-pipeline metadata.

    With ``name=None`` a dictionary for every known backend is returned. An
    explicit backend name returns one dictionary. This function reports
    software support boundaries only; it does not select or execute a backend.
    """

    def one(backend_name: str):
        if backend_name not in _BACKEND_METADATA:
            raise ValueError("Unknown backend name.")
        available = (
            True
            if backend_name == "numpy"
            else has_numba()
            if backend_name == "numba"
            else has_jax()
        )
        return {
            "name": backend_name,
            **_BACKEND_METADATA[backend_name],
            "available": bool(available),
        }

    if name is None:
        return {backend_name: one(backend_name) for backend_name in _BACKEND_METADATA}
    return one(str(name).lower().strip())


def get_backend_name(
    preferred: Optional[str] = None,
    *,
    auto: bool = False,
    prefer_gpu: bool = False,
) -> BackendName:
    """Resolve an installed primitive backend without changing canonical scope.

    ``None`` resolves to NumPy. ``auto``/``best`` chooses an installed optional
    primitive backend according to the historical preference order. Explicit
    unavailable requests fail rather than silently falling back.
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


def select_backend(
    preferred: Optional[str] = None,
    *,
    auto: bool = False,
    prefer_gpu: bool = False,
):
    """Return one lightweight primitive-backend mapping with capability metadata."""
    name = get_backend_name(preferred, auto=auto, prefer_gpu=prefer_gpu)
    metadata = backend_capabilities(name)

    if name == "numpy":
        return {
            **metadata,
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
            **metadata,
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
            **metadata,
            "normalize": normalize_jax,
            "central_difference": central_difference_jax,
            "apply_window": apply_window_jax,
            "causal_moving_correlation": causal_moving_correlation_jax,
        }

    raise BackendUnavailableError(f"Unsupported backend: {name}")


def get_backend(
    preferred: Optional[str] = None,
    *,
    auto: bool = False,
    prefer_gpu: bool = False,
):
    """Compatibility alias for :func:`select_backend`."""
    return select_backend(preferred, auto=auto, prefer_gpu=prefer_gpu)
