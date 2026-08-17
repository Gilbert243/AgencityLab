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


def _default_speed_tolerance(beta: np.ndarray, axis: np.ndarray) -> float:
    """Return a scale-aware floating-point zero threshold for ``|beta_dot|``."""

    if beta.size < 2:
        return 0.0
    scale = max(
        float(np.max(np.abs(beta))),
        float(np.ptp(np.real(beta))),
        float(np.ptp(np.imag(beta))),
    )
    if scale == 0.0:
        return 0.0
    min_step = float(np.min(np.diff(axis)))
    return float(64.0 * np.finfo(float).eps * scale / min_step)


def curvature(beta, xi=None, *, speed_tolerance: float | None = None) -> np.ndarray:
    """Approximate the signed algebraic curvature of ``beta(t)``.

    The canonical trajectory is not changed.  Curvature is returned as ``NaN``
    wherever the discrete velocity is numerically unresolved.  The default
    threshold is scale-aware floating-point zero detection, not a physical
    epsilon.  Callers may provide an explicit non-negative numerical tolerance.
    """

    beta = _complex_1d(beta, name="beta")
    if beta.size < 3:
        return np.full(beta.size, np.nan, dtype=float)
    axis = _axis(xi, beta.size)
    d1 = np.gradient(beta, axis, edge_order=2)
    d2 = np.gradient(d1, axis, edge_order=2)
    speed = np.abs(d1)

    if speed_tolerance is None:
        tolerance = _default_speed_tolerance(beta, axis)
    else:
        try:
            tolerance = float(speed_tolerance)
        except Exception as exc:
            raise ValueError("speed_tolerance must be finite and non-negative") from exc
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("speed_tolerance must be finite and non-negative")

    out = np.full(beta.size, np.nan, dtype=float)
    defined = speed > tolerance
    numerator = np.imag(np.conjugate(d1[defined]) * d2[defined])
    out[defined] = numerator / speed[defined] ** 3
    return out


def radius(beta) -> np.ndarray:
    """Return ``|beta|``."""
    return np.abs(_complex_1d(beta, name="beta"))


def net_phase_turns(theta, *, valid_mask=None) -> float:
    """Return net unwrapped structural-orientation turns on any valid interval."""

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


def winding_number(theta, *, valid_mask=None, closed: bool = True) -> float:
    """Return winding only when the supplied interval is declared closed.

    ``closed=True`` preserves the historical direct-call behaviour.  Reports
    should use :func:`net_phase_turns` for open finite records and reserve this
    name for a complete closed period/contour.
    """

    if not isinstance(closed, (bool, np.bool_)):
        raise ValueError("closed must be Boolean")
    if not closed:
        return float("nan")
    return net_phase_turns(theta, valid_mask=valid_mask)


def winding_diagnostic(
    theta,
    *,
    valid_mask=None,
    closed: bool = True,
) -> dict[str, float | int | bool | str]:
    """Return winding diagnostics with an explicit closure declaration."""

    turns = net_phase_turns(theta, valid_mask=valid_mask)
    value = winding_number(theta, valid_mask=valid_mask, closed=closed)
    if not np.isfinite(value):
        return {
            "defined": False,
            "closure_declared": bool(closed),
            "net_phase_turns": turns,
            "winding_number": float("nan"),
            "nearest_integer": 0,
            "integer_residual": float("nan"),
            "reason": "interval not closed or structural orientation undefined",
        }
    nearest = int(np.rint(value))
    return {
        "defined": True,
        "closure_declared": True,
        "net_phase_turns": turns,
        "winding_number": value,
        "nearest_integer": nearest,
        "integer_residual": float(abs(value - nearest)),
        "reason": "closed interval declared by caller",
    }


def geometric_summary(
    beta,
    *,
    xi=None,
    theta=None,
    valid_mask=None,
    winding_closed: bool = True,
) -> dict[str, object]:
    """Return beta-trajectory geometry and explicit open/closed phase diagnostics."""

    beta = _complex_1d(beta, name="beta")
    kappa = curvature(beta, xi=xi)
    finite = np.isfinite(kappa)
    summary: dict[str, object] = {
        "radius_mean": float(np.mean(np.abs(beta))) if beta.size else float("nan"),
        "radius_std": float(np.std(np.abs(beta))) if beta.size else float("nan"),
        "trajectory_length": trajectory_length(beta),
        "curvature": kappa,
        "curvature_mean": float(np.mean(kappa[finite])) if np.any(finite) else float("nan"),
        "curvature_mean_abs": (
            float(np.mean(np.abs(kappa[finite])))
            if np.any(finite)
            else float("nan")
        ),
        "curvature_std": float(np.std(kappa[finite])) if np.any(finite) else float("nan"),
        "curvature_defined_fraction": float(np.mean(finite)) if kappa.size else 0.0,
        "geometry_source": "beta",
    }
    if theta is not None:
        summary["net_phase_turns"] = net_phase_turns(theta, valid_mask=valid_mask)
        summary["winding"] = winding_diagnostic(
            theta,
            valid_mask=valid_mask,
            closed=winding_closed,
        )
    return summary
