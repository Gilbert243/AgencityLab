"""Two-dimensional U(1) vortex reference structures.

Volume 2, Eq. (17.4), uses the normalized ansatz
``psi(r, theta) = f(r) exp(i*n*theta)`` with integer winding ``n``. The source
provides the radial differential equation and boundary conditions, but not a
closed-form exact radial profile. Accordingly, this module requires callers to
supply ``f(r)`` and provides a numerical residual evaluator rather than
inventing an analytic solution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from agencitylab.fields.physics import vacuum_amplitude
from agencitylab.scientific_status import ScientificStatus

SCIENTIFIC_STATUS = ScientificStatus.RESEARCH


def _finite_real_scalar(value, *, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_winding(winding) -> int:
    if isinstance(winding, (bool, np.bool_)) or not isinstance(winding, (int, np.integer)):
        raise TypeError("winding must be an integer")
    return int(winding)


def _finite_real_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(array, dtype=float)
    except Exception as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _polar_coordinates(*, x=None, y=None, r=None, theta=None) -> tuple[np.ndarray, np.ndarray]:
    cartesian = x is not None or y is not None
    polar = r is not None or theta is not None
    if cartesian and polar:
        raise ValueError("provide either x/y coordinates or r/theta coordinates, not both")

    if polar:
        if r is None or theta is None:
            raise ValueError("both r and theta are required for explicit polar coordinates")
        radius = _finite_real_array(r, name="r")
        angle = _finite_real_array(theta, name="theta")
        if radius.shape != angle.shape:
            raise ValueError("r and theta must have the same shape")
        if np.any(radius < 0.0):
            raise ValueError("r must be non-negative")
        return radius, angle

    if x is None or y is None:
        raise ValueError("provide x and y, or provide r and theta")
    x_values = _finite_real_array(x, name="x")
    y_values = _finite_real_array(y, name="y")

    if x_values.ndim == 1 and y_values.ndim == 1:
        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="ij")
    elif x_values.shape == y_values.shape and x_values.ndim == 2:
        x_grid, y_grid = x_values, y_values
    else:
        raise ValueError("x/y must be one-dimensional axes or matching two-dimensional grids")

    radius = np.hypot(x_grid, y_grid)
    angle = np.arctan2(y_grid, x_grid)
    return radius, angle


def vortex_field(
    *,
    winding: int,
    lambda_: float,
    mu: float,
    radial_profile: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    x=None,
    y=None,
    r=None,
    theta=None,
) -> np.ndarray:
    """Construct a two-dimensional U(1) vortex reference from a supplied radial profile.

    The returned field is

    ``phi = sqrt(lambda/mu) * f(r) * exp(i*n*theta)``.

    ``lambda_ > 0`` and ``mu > 0`` are required because the vortex is defined
    around the broken-symmetry U(1) vacuum. ``winding`` must be an integer.

    Coordinates may be supplied either as one-dimensional ``x``/``y`` axes,
    matching two-dimensional Cartesian grids, or explicit matching ``r`` and
    ``theta`` arrays. The ambient reference structure is two-dimensional; no
    artificial N-D vortex generalization is implied.

    ``radial_profile`` must be caller supplied, either as an array matching the
    resolved radius shape or a callable evaluated on that radius. This API
    intentionally provides no built-in fake exact profile. For non-zero
    winding, an exact grid point at ``r == 0`` must have ``f(0) == 0``.
    """

    n = _validate_winding(winding)
    amplitude = vacuum_amplitude(lambda_, mu)
    radius, angle = _polar_coordinates(x=x, y=y, r=r, theta=theta)

    if callable(radial_profile):
        profile = radial_profile(radius)
    else:
        profile = radial_profile
    profile_array = _finite_real_array(profile, name="radial_profile")
    if profile_array.shape != radius.shape:
        raise ValueError("radial_profile must match the resolved coordinate shape")

    if n != 0:
        core = radius == 0.0
        if np.any(profile_array[core] != 0.0):
            raise ValueError("non-zero winding requires f(0) == 0 at exact core points")

    field = amplitude * profile_array * np.exp(1j * n * angle)
    if n != 0:
        field = np.asarray(field, dtype=complex)
        field[radius == 0.0] = 0.0j
    return field


def vortex_radial_residual(
    r,
    radial_profile,
    *,
    winding: int,
    lambda_: float,
) -> np.ndarray:
    """Evaluate the interior residual of the physical radial vortex equation.

    Starting from the static quartic equation

    ``-laplacian(phi) - lambda*phi + mu*|phi|^2*phi = 0``

    and writing ``phi = sqrt(lambda/mu) f(r) exp(i*n*theta)``, the amplitude
    factor cancels and the radial equation is

    ``-f'' - f'/r + n^2*f/r^2 - lambda*f + lambda*f^3 = 0``.

    This general ``lambda`` form is a mathematical rescaling/consequence of
    the normalized Volume-2 Eq. (17.4), not a separate source equation. ``mu``
    does not appear because it cancels after factoring out the vacuum amplitude.

    The returned array contains the residual at ``r[1:-1]`` only. This avoids
    pretending that the singular-looking coordinate terms at ``r = 0`` can be
    evaluated by the same stencil. Derivatives use NumPy's second-order
    ``gradient`` formulas on the caller-provided one-dimensional radial grid.
    No exact vortex profile or universal core scale is assumed.
    """

    n = _validate_winding(winding)
    radius = _finite_real_array(r, name="r")
    profile = _finite_real_array(radial_profile, name="radial_profile")
    if radius.ndim != 1 or profile.ndim != 1:
        raise ValueError("r and radial_profile must be one-dimensional")
    if radius.shape != profile.shape:
        raise ValueError("r and radial_profile must have the same shape")
    if radius.size < 4:
        raise ValueError("at least four radial points are required")
    differences = np.diff(radius)
    if np.any(differences <= 0.0):
        raise ValueError("r must be strictly increasing")
    if radius[0] < 0.0:
        raise ValueError("r must be non-negative")
    if np.any(radius[1:-1] <= 0.0):
        raise ValueError("interior radial points must be strictly positive")

    lambda_value = _finite_real_scalar(lambda_, name="lambda_")
    if lambda_value <= 0.0:
        raise ValueError("vortex broken vacuum requires lambda_ > 0")

    first = np.gradient(profile, radius, edge_order=2)
    second = np.gradient(first, radius, edge_order=2)
    interior_r = radius[1:-1]
    interior_f = profile[1:-1]
    return (
        -second[1:-1]
        - first[1:-1] / interior_r
        + (n * n) * interior_f / (interior_r * interior_r)
        - lambda_value * interior_f
        + lambda_value * interior_f**3
    )
