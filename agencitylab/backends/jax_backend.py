"""Optional JAX-backed primitives for AgencityLab experiments.

These helpers support research on autodiff, vectorisation, and accelerator
execution. They are not the canonical public pipeline, which remains the
float64-oriented NumPy reference. Direct JAX users must therefore validate
precision and device behaviour for their own workload.
"""

from __future__ import annotations


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
    """Apply an optional diagnostic normalisation with JAX."""
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
    """Compute a JAX-compatible central difference for a one-dimensional array."""
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
    """Apply a JAX-compatible tapering window to one-dimensional data."""
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
    """Compute experimental JAX CRM with exact zero-variance branching.

    ``epsilon`` remains in the signature for source compatibility but is not
    inserted into the Pearson denominator and does not define physical zero.
    """
    del epsilon
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
        ss_a = jnp.dot(a0, a0)
        ss_b = jnp.dot(b0, b0)

        def zero(_):
            return jnp.asarray(0.0, dtype=values.dtype)

        def nonzero(_):
            corr = jnp.dot(a0, b0) / (jnp.sqrt(ss_a) * jnp.sqrt(ss_b))
            return jnp.clip(corr, -1.0, 1.0)

        return lax.cond((ss_a == 0.0) | (ss_b == 0.0), zero, nonzero, operand=None)

    vals = jax.vmap(corr_at)(idx)
    out = jnp.zeros(n, dtype=values.dtype)
    return out.at[idx].set(vals)
