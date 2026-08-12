"""Intrinsic kinematic primitives from Volume 2 Definition 12.4.

The source defines, for a curve on a Riemannian manifold, velocity ``X``,
covariant acceleration ``A = nabla_X X``, and

    D = sqrt(g(X, X) + g(A, X)^2).

The detailed Riemannian Agencity analysis is explicitly deferred by Volume 2.
Accordingly this module accepts caller-supplied tangent vectors, covariant
acceleration, and metric tensors; it does not invent a Levi-Civita solver,
coordinate chart system, or full CRM pipeline on manifolds.

Scientific status: experimental mathematical extension.
"""

from __future__ import annotations

import numpy as np

from agencitylab.scientific_status import ScientificStatus

RIEMANNIAN_EXTENSION_STATUS = ScientificStatus.EXPERIMENTAL


def _validate_metric(metric, vector_shape: tuple[int, ...]) -> np.ndarray:
    tensor = np.asarray(metric, dtype=float)
    dimension = vector_shape[-1]
    expected = vector_shape[:-1] + (dimension, dimension)
    if tensor.shape != expected:
        raise ValueError(f"metric must have shape {expected}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("metric must contain only finite values")
    if not np.allclose(tensor, np.swapaxes(tensor, -1, -2), rtol=1e-12, atol=1e-14):
        raise ValueError("metric must be symmetric")

    flat = tensor.reshape((-1, dimension, dimension))
    for item in flat:
        try:
            np.linalg.cholesky(item)
        except np.linalg.LinAlgError as exc:
            raise ValueError("metric must be positive definite") from exc
    return tensor


def _validate_vector(value, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim < 1 or vector.shape[-1] < 1:
        raise ValueError(f"{name} must have a non-empty final coordinate axis")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def riemannian_inner_product(vector_a, vector_b, metric) -> np.ndarray:
    """Return ``g(vector_a, vector_b)`` pointwise."""

    first = _validate_vector(vector_a, name="vector_a")
    second = _validate_vector(vector_b, name="vector_b")
    if first.shape != second.shape:
        raise ValueError("vector_a and vector_b must have identical shapes")
    tensor = _validate_metric(metric, first.shape)
    return np.einsum("...i,...ij,...j->...", first, tensor, second)


def riemannian_speed(velocity, metric) -> np.ndarray:
    """Return ``||X||_g = sqrt(g(X,X))`` from Definition 12.4."""

    vector = _validate_vector(velocity, name="velocity")
    norm_squared = riemannian_inner_product(vector, vector, metric)
    if np.any(norm_squared < 0.0):
        raise ValueError("Riemannian norm squared must be non-negative")
    return np.sqrt(norm_squared)


def riemannian_dynamic_intensity(velocity, covariant_acceleration, metric) -> np.ndarray:
    """Return the intrinsic Volume-2 dynamic intensity on a manifold.

    Implements ``sqrt(||X||_g^2 + g(A,X)^2)``.  ``covariant_acceleration``
    must already represent ``nabla_X X`` in the caller's coordinate system.
    """

    vector = _validate_vector(velocity, name="velocity")
    acceleration = _validate_vector(
        covariant_acceleration,
        name="covariant_acceleration",
    )
    if vector.shape != acceleration.shape:
        raise ValueError("velocity and covariant_acceleration must have identical shapes")
    tensor = _validate_metric(metric, vector.shape)
    speed_squared = riemannian_inner_product(vector, vector, tensor)
    acceleration_projection = riemannian_inner_product(acceleration, vector, tensor)
    radicand = speed_squared + acceleration_projection**2
    if np.any(radicand < 0.0):
        raise ValueError("Riemannian dynamic-intensity radicand must be non-negative")
    return np.sqrt(radicand)
