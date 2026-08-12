"""One-dimensional real-sector domain-wall reference.

Volume 2, Eq. (17.3), gives the normalized exact kink
``psi(x) = tanh((x - x0) / sqrt(2))`` for the real restriction of the
normalized static phi^4 equation. For the already-implemented quartic
potential

``V(phi) = -(lambda/2)|phi|^2 + (mu/4)|phi|^4``,

the profile returned here,

``phi(x) = v * tanh(sqrt(lambda/2) * (x - x0))``,
``v = sqrt(lambda/mu)``,

is a mathematical consequence obtained by rescaling the source equation. It
is not presented as a second independent source formula.

Scientific limitation
---------------------
The full complex potential has a connected U(1) vacuum manifold. The kink is
therefore exposed only as a real-sector / Z2 reference solution. It is not a
generally stable topological wall of the full complex U(1) theory.
"""

from __future__ import annotations

import numpy as np

from agencitylab.fields.numerics import NeumannBoundary, UniformRectilinearGrid, laplacian
from agencitylab.fields.physics import QuarticAgencityPotential, vacuum_amplitude
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


def _validate_real_1d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real for the real-sector domain wall")
    try:
        real = np.asarray(array, dtype=float)
    except Exception as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(real)):
        raise ValueError(f"{name} must contain only finite values")
    return real


def domain_wall_profile(
    x,
    *,
    lambda_: float,
    mu: float,
    center: float = 0.0,
    orientation: int = 1,
) -> np.ndarray:
    """Return the rescaled real phi^4 kink on a one-dimensional coordinate array.

    Parameters
    ----------
    x:
        One-dimensional finite spatial coordinates.
    lambda_, mu:
        Quartic-potential parameters. Both must be finite and strictly positive
        because this reference wall connects the two real broken vacua
        ``-sqrt(lambda/mu)`` and ``+sqrt(lambda/mu)``.
    center:
        Wall centre ``x0``.
    orientation:
        ``+1`` for the kink and ``-1`` for the anti-kink. No arbitrary complex
        phase is introduced: this function is intentionally real-sector only.

    Returns
    -------
    numpy.ndarray
        Real array with the same shape as ``x``.

    Notes
    -----
    The normalized source profile is Volume-2 Eq. (17.3). The general
    ``lambda_, mu`` form is a mathematical rescaling of that equation.
    Scientific status: ``research``.
    """

    coordinates = _validate_real_1d(x, name="x")
    center_value = _finite_real_scalar(center, name="center")
    if isinstance(orientation, (bool, np.bool_)) or orientation not in (-1, 1):
        raise ValueError("orientation must be exactly +1 or -1")

    amplitude = vacuum_amplitude(lambda_, mu)
    lambda_value = _finite_real_scalar(lambda_, name="lambda_")
    scale = np.sqrt(lambda_value / 2.0)
    return orientation * amplitude * np.tanh(scale * (coordinates - center_value))


def domain_wall_residual(
    phi,
    grid: UniformRectilinearGrid,
    *,
    lambda_: float,
    mu: float,
    boundary=None,
) -> np.ndarray:
    """Evaluate the discrete static real-sector wall equation residual.

    The evaluated equation is

    ``phi_xx + lambda*phi - mu*phi**3 = 0``.

    This is implemented as ``laplacian(phi) - potential.gradient(phi)`` so the
    quartic physics formula remains owned by ``agencitylab.fields.physics``.
    The numerical Laplacian is the reusable second-order operator from
    ``agencitylab.fields.numerics``.

    ``boundary`` is passed explicitly to the numerical operator. When omitted,
    homogeneous Neumann data are used as a practical reference convention for
    a sufficiently large truncated wall domain. Boundary points can therefore
    carry truncation error; convergence tests should assess an interior region.
    No epsilon or fitted correction is added to improve the residual.
    """

    if not isinstance(grid, UniformRectilinearGrid):
        raise TypeError("grid must be UniformRectilinearGrid")
    if grid.ndim != 1:
        raise ValueError("domain_wall_residual requires a one-dimensional grid")

    field = _validate_real_1d(phi, name="phi")
    if field.shape != grid.shape:
        raise ValueError(f"phi shape {field.shape} does not match grid shape {grid.shape}")

    potential = QuarticAgencityPotential(lambda_=lambda_, mu=mu)
    if potential.lambda_ <= 0.0:
        raise ValueError("domain-wall broken vacua require lambda_ > 0")

    resolved_boundary = NeumannBoundary(gradient=0.0) if boundary is None else boundary
    return laplacian(field, grid, boundary=resolved_boundary) - potential.gradient(field)
