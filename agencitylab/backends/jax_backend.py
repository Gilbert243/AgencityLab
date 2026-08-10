"""
JAX-backed accelerations for AgencityLab.

This module is optional and designed for research workflows where autodiff,
vectorization or accelerator execution may be useful.

JAX can run on GPU/TPU automatically when the proper backend is installed.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def _require_jax():
    """Import JAX lazily."""
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        from jax import lax  # type: ignore
        return jax, jnp, lax
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
    _, jnp, _ = _require_jax()
    u = jnp.asarray(u, dtype=jnp.float32)

    method = str(method).lower().strip()

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


def central_difference_jax(values, step: float):
    """
    JAX-compatible central difference for a 1D array.
    """
    _, jnp, _ = _require_jax()
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)

    if values.size < 2:
        raise ValueError("values must contain at least two samples.")
    if step <= 0:
        raise ValueError("step must be positive.")

    out = jnp.zeros_like(values)
    out = out.at[1:-1].set((values[2:] - values[:-2]) / (2.0 * step))
    out = out.at[0].set((values[1] - values[0]) / step)
    out = out.at[-1].set((values[-1] - values[-2]) / step)
    return out


def apply_window_jax(values, kind: str = "hann"):
    """
    JAX-compatible windowing for a 1D array.
    """
    _, jnp, _ = _require_jax()
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    kind = str(kind).lower().strip()
    n = values.shape[0]

    if kind == "hann":
        window = jnp.hanning(n)
    elif kind == "hamming":
        window = jnp.hamming(n)
    elif kind == "blackman":
        window = jnp.blackman(n)
    elif kind == "rectangular":
        window = jnp.ones(n, dtype=values.dtype)
    else:
        raise ValueError("Unknown window kind.")

    return values * window


def causal_moving_correlation_jax(values, window: int = 1, epsilon: float = 1e-12):
    """
    JAX-compatible causal moving correlation for a 1D signal.
    """
    jax, jnp, lax = _require_jax()
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    n = values.shape[0]

    if window < 1:
        raise ValueError("window must be >= 1.")
    if n < 2 * window:
        raise ValueError("values must contain at least 2*window samples.")

    idx = jnp.arange(2 * window - 1, n)

    def corr_at(i):
        a = lax.dynamic_slice(values, (i - 2 * window + 1,), (window,))
        b = lax.dynamic_slice(values, (i - window + 1,), (window,))

        a0 = a - jnp.mean(a)
        b0 = b - jnp.mean(b)

        denom = jnp.linalg.norm(a0) * jnp.linalg.norm(b0)
        corr = jnp.where(denom < epsilon, 0.0, jnp.dot(a0, b0) / (denom + epsilon))
        return jnp.clip(corr, -1.0, 1.0)

    vals = jax.vmap(corr_at)(idx)

    out = jnp.zeros(n, dtype=values.dtype)
    out = out.at[idx].set(vals)
    return out