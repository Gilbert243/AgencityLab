"""Geometry of the intrinsic agencity-state trajectory ``beta(t)``.

The theoretical geometric observables are defined on beta, not on b.  This
matters when characteristic power varies in time: multiplication by ``P_c(t)``
can change the geometry of the observable flux without changing the intrinsic
state trajectory.
"""

from __future__ import annotations

import numpy as np


def _complex_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=complex)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _real_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _axis(xi, n: int) -> np.ndarray:
    if xi is None:
        return np.arange(n, dtype=float)
    axis = _real_1d(xi, name="xi")
    if axis.size != n:
        raise ValueError("xi must match the trajectory length")
    if axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("xi must be strictly increasing")
    return axis


def compute_angle(b, *, unwrap: bool = True) -> np.ndarray:
    """Return the phase of a complex trajectory.

    This compatibility helper describes the phase of its input.  Structural
    orientation should be taken from canonical ``Theta``, not inferred from
    ``beta`` or ``b`` when J may change sign.
    """
    values = _complex_1d(b, name="b")
    theta = np.angle(values)
    return np.unwrap(theta) if unwrap else theta


def trajectory_length(beta) -> float:
    """Return the polygonal arc length of the beta trajectory."""
    beta = _complex_1d(beta, name="beta")
    if beta.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(beta))))


def curvature(beta, xi=None) -> np.ndarray:
    """Approximate the theory's signed algebraic curvature of ``beta(t)``.

    The continuous definition is

        kappa = Im(conj(beta_dot) * beta_ddot) / |beta_dot|**3

    wherever ``beta_dot != 0``.  Samples where the discrete velocity is exactly
    zero are returned as ``NaN`` because curvature is undefined there.  No
    epsilon is inserted into the denominator.
    """
    beta = _complex_1d(beta, name="beta")
    if beta.size < 3:
        return np.full(beta.size, np.nan, dtype=float)
    axis = _axis(xi, beta.size)
    edge_order = 2 if beta.size > 2 else 1
    d1 = np.gradient(beta, axis, edge_order=edge_order)
    d2 = np.gradient(d1, axis, edge_order=edge_order)
    speed = np.abs(d1)
    out = np.full(beta.size, np.nan, dtype=float)
    defined = speed > 0.0
    out[defined] = np.imag(np.conjugate(d1[defined]) * d2[defined]) / speed[defined] ** 3
    return out


def radius(beta) -> np.ndarray:
    """Return ``|beta|``."""
    return np.abs(_complex_1d(beta, name="beta"))


def winding_number(theta, *, valid_mask=None) -> float:
    """Return the net structural-orientation winding over the supplied interval.

    This is the discrete approximation of

        W = (1 / 2pi) integral Theta_dot dt.

    The theoretical integer interpretation applies to a complete closed period.
    If structural orientation is undefined at any supplied sample, the global
    winding is returned as ``NaN`` rather than bridging that topological gap.
    """
    theta = _real_1d(theta, name="theta")
    if theta.size < 2:
        return float("nan")
    if valid_mask is None:
        valid = np.ones(theta.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim != 1 or valid.size != theta.size:
            raise ValueError("valid_mask must match theta")
    if not np.all(valid):
        return float("nan")
    unwrapped = np.unwrap(theta)
    return float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))


def winding_diagnostic(theta, *, valid_mask=None) -> dict[str, float | int | bool]:
    """Return winding value and its distance from the nearest integer."""
    value = winding_number(theta, valid_mask=valid_mask)
    if not np.isfinite(value):
        return {
            "defined": False,
            "winding_number": float("nan"),
            "nearest_integer": 0,
            "integer_residual": float("nan"),
        }
    nearest = int(np.rint(value))
    return {
        "defined": True,
        "winding_number": value,
        "nearest_integer": nearest,
        "integer_residual": float(abs(value - nearest)),
    }


def geometric_summary(beta, *, xi=None, theta=None, valid_mask=None) -> dict[str, object]:
    """Return theory-facing beta-trajectory geometry and optional winding."""
    beta = _complex_1d(beta, name="beta")
    kappa = curvature(beta, xi=xi)
    finite = np.isfinite(kappa)
    summary: dict[str, object] = {
        "radius_mean": float(np.mean(np.abs(beta))) if beta.size else float("nan"),
        "radius_std": float(np.std(np.abs(beta))) if beta.size else float("nan"),
        "trajectory_length": trajectory_length(beta),
        "curvature": kappa,
        "curvature_mean": float(np.mean(kappa[finite])) if np.any(finite) else float("nan"),
        "curvature_mean_abs": float(np.mean(np.abs(kappa[finite]))) if np.any(finite) else float("nan"),
        "curvature_std": float(np.std(kappa[finite])) if np.any(finite) else float("nan"),
        "curvature_defined_fraction": float(np.mean(finite)) if kappa.size else 0.0,
        "geometry_source": "beta",
    }
    if theta is not None:
        summary["winding"] = winding_diagnostic(theta, valid_mask=valid_mask)
    return summary
