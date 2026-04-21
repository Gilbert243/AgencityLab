"""
JAX-backed accelerations for AgencityLab.

This module is optional and designed for research workflows where autodiff,
vectorization or accelerator execution may be useful.
"""

from __future__ import annotations

from typing import Any


def _require_jax():
    """Import JAX lazily."""
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        return jax, jnp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "JAX is not installed. Install AgencityLab with the jax extra."
        ) from exc


def has_jax() -> bool:
    """Return True if JAX is available."""
    try:
        _require_jax()
        return True
    except Exception:
        return False


def normalize_jax(u, method: str = "zscore", epsilon: float = 1e-12):
    """
    JAX-compatible normalization helper.
    """
    _, jnp = _require_jax()
    u = jnp.asarray(u, dtype=jnp.float32)

    if method == "zscore":
        mean = jnp.mean(u)
        std = jnp.std(u)
        return jnp.where(std < epsilon, jnp.zeros_like(u), (u - mean) / std)

    if method == "minmax":
        u_min = jnp.min(u)
        u_max = jnp.max(u)
        span = u_max - u_min
        return jnp.where(span < epsilon, jnp.zeros_like(u), (u - u_min) / span)

    if method == "centered":
        return u - jnp.mean(u)

    raise ValueError("Unknown normalization method.")
