from __future__ import annotations

import numpy as np
import pytest

from agencitylab.fields.coherent import (
    coherence_length,
    dimensionless_effective_potential,
    dimensionless_static_residual,
    from_dimensionless_field,
    to_dimensionless_field,
)
from agencitylab.fields.numerics import UniformRectilinearGrid


def test_chapter17_rescaling_round_trip() -> None:
    phi = np.array([0.2 + 0.4j, -0.6j])
    psi = to_dimensionless_field(phi, lambda_=2.0, mu=0.5)
    restored = from_dimensionless_field(psi, lambda_=2.0, mu=0.5)
    np.testing.assert_allclose(restored, phi)
    assert coherence_length(4.0) == pytest.approx(0.5)


def test_dimensionless_potential_has_u1_vacuum_manifold() -> None:
    phases = np.linspace(-np.pi, np.pi, 9)
    vacuum = np.exp(1j * phases)
    np.testing.assert_allclose(dimensionless_effective_potential(vacuum), 0.0, atol=1e-31)
    assert dimensionless_effective_potential(np.array([0.0]))[0] == pytest.approx(0.25)


def test_uniform_dimensionless_vacuum_solves_static_equation() -> None:
    grid = UniformRectilinearGrid(shape=(12,), spacings=(0.2,))
    psi = np.full(grid.shape, np.exp(0.37j))
    residual = dimensionless_static_residual(psi, grid, boundary="periodic")
    np.testing.assert_allclose(residual, 0.0, atol=1e-14)


def test_dimensionless_rescaling_requires_broken_phase_parameters() -> None:
    with pytest.raises(ValueError):
        coherence_length(0.0)
    with pytest.raises(ValueError):
        to_dimensionless_field(np.ones(2), lambda_=-1.0, mu=1.0)
    with pytest.raises(ValueError):
        from_dimensionless_field(np.ones(2), lambda_=1.0, mu=0.0)
