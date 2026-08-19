from __future__ import annotations

import numpy as np
import pytest

import agencitylab
import agencitylab.fields as fields_api
from agencitylab.fields.dynamics import (
    SCIENTIFIC_STATUS,
    dissipative_klein_gordon_acceleration,
    klein_gordon_acceleration,
    simulate_tdgl,
    tdgl_rhs,
)
from agencitylab.fields.numerics import (
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
    UniformRectilinearGrid,
)
from agencitylab.fields.physics import QuarticAgencityPotential
from agencitylab.scientific_status import ScientificStatus


def _periodic_grid(n: int = 32) -> UniformRectilinearGrid:
    return UniformRectilinearGrid(shape=(n,), spacings=(2.0 * np.pi / n,))


def test_dynamics_status_and_public_boundary_are_research_only():
    assert SCIENTIFIC_STATUS is ScientificStatus.RESEARCH
    assert agencitylab.__version__ == "1.1.3"
    assert fields_api.simulate_klein_gordon is not None
    assert not hasattr(agencitylab, "simulate_klein_gordon")


@pytest.mark.parametrize("theta", [0.0, 0.7])
def test_homogeneous_broken_vacuum_is_stationary_for_all_boundaries(theta):
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.25,))
    potential = QuarticAgencityPotential(lambda_=2.0, mu=8.0)
    vacuum = np.sqrt(potential.lambda_ / potential.mu) * np.exp(1j * theta)
    phi = np.full(grid.shape, vacuum, dtype=complex)

    boundaries = (
        PeriodicBoundary(),
        NeumannBoundary(gradient=0.0),
        DirichletBoundary(value=vacuum),
    )
    for boundary in boundaries:
        acceleration = klein_gordon_acceleration(
            phi,
            grid,
            potential,
            boundary=boundary,
        )
        rhs = tdgl_rhs(phi, grid, potential, gamma=1.3, boundary=boundary)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-12)
        np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_origin_is_exact_stationary_state():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    potential = QuarticAgencityPotential(lambda_=1.0, mu=2.0)
    phi = np.zeros(grid.shape, dtype=complex)

    np.testing.assert_array_equal(
        klein_gordon_acceleration(phi, grid, potential),
        np.zeros_like(phi),
    )
    np.testing.assert_array_equal(
        dissipative_klein_gordon_acceleration(
            phi,
            np.zeros_like(phi),
            grid,
            potential,
            gamma=0.4,
        ),
        np.zeros_like(phi),
    )
    np.testing.assert_array_equal(
        tdgl_rhs(phi, grid, potential, gamma=0.4),
        np.zeros_like(phi),
    )


@pytest.mark.parametrize("complex_field", [False, True])
def test_primitives_support_real_complex_and_two_dimensional_fields(complex_field):
    grid = UniformRectilinearGrid(shape=(6, 5), spacings=(0.3, 0.4))
    x, y = grid.axes
    phi = np.sin(x)[:, None] + 0.4 * np.cos(y)[None, :]
    if complex_field:
        phi = phi + 0.2j * np.cos(x)[:, None]
    phi_dot = 0.1 * phi
    potential = QuarticAgencityPotential(lambda_=-0.5, mu=1.25)

    conservative = klein_gordon_acceleration(phi, grid, potential)
    dissipative = dissipative_klein_gordon_acceleration(
        phi,
        phi_dot,
        grid,
        potential,
        gamma=0.7,
    )
    overdamped = tdgl_rhs(phi, grid, potential, gamma=0.7)

    assert conservative.shape == grid.shape
    assert dissipative.shape == grid.shape
    assert overdamped.shape == grid.shape
    assert np.all(np.isfinite(conservative))
    assert np.all(np.isfinite(dissipative))
    assert np.all(np.isfinite(overdamped))
    assert np.iscomplexobj(conservative) is complex_field


@pytest.mark.parametrize("lambda_", [1.0, 0.0, -1.0])
def test_lambda_signs_allowed_by_existing_potential_work_in_dynamics(lambda_):
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi = np.linspace(-0.2, 0.3, 8)
    potential = QuarticAgencityPotential(lambda_=lambda_, mu=1.0)
    result = klein_gordon_acceleration(phi, grid, potential)
    assert result.shape == grid.shape
    assert np.all(np.isfinite(result))


def test_zero_gamma_dissipative_primitive_is_exactly_conservative():
    grid = _periodic_grid(16)
    x = grid.axes[0]
    phi = 0.2 * np.exp(2j * x)
    phi_dot = (0.1 - 0.05j) * phi
    potential = QuarticAgencityPotential(lambda_=0.75, mu=1.5)

    conservative = klein_gordon_acceleration(phi, grid, potential)
    dissipative = dissipative_klein_gordon_acceleration(
        phi,
        phi_dot,
        grid,
        potential,
        gamma=0.0,
    )
    np.testing.assert_array_equal(dissipative, conservative)


@pytest.mark.parametrize("gamma", [-1.0, -1e-12, np.nan, np.inf, -np.inf])
def test_dissipative_klein_gordon_rejects_invalid_gamma(gamma):
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi = np.zeros(grid.shape)
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    with pytest.raises(ValueError):
        dissipative_klein_gordon_acceleration(
            phi,
            phi,
            grid,
            potential,
            gamma=gamma,
        )


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_tdgl_requires_finite_strictly_positive_gamma(gamma):
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi = np.zeros(grid.shape)
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    with pytest.raises(ValueError):
        tdgl_rhs(phi, grid, potential, gamma=gamma)


@pytest.mark.parametrize(
    "bad_phi",
    [
        np.array([0.0, 1.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_primitives_reject_nan_and_inf_fields(bad_phi):
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    with pytest.raises(ValueError):
        klein_gordon_acceleration(bad_phi, grid, potential)
    with pytest.raises(ValueError):
        tdgl_rhs(bad_phi, grid, potential, gamma=1.0)


def test_dissipative_primitive_rejects_bad_velocity_and_shape():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi = np.zeros(grid.shape)
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)

    with pytest.raises(ValueError):
        dissipative_klein_gordon_acceleration(
            phi,
            np.zeros(7),
            grid,
            potential,
            gamma=0.0,
        )
    bad_velocity = np.zeros(grid.shape)
    bad_velocity[2] = np.nan
    with pytest.raises(ValueError):
        dissipative_klein_gordon_acceleration(
            phi,
            bad_velocity,
            grid,
            potential,
            gamma=0.0,
        )


def test_grid_shape_mismatch_is_rejected():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    with pytest.raises(ValueError, match="grid shape"):
        klein_gordon_acceleration(np.zeros(7), grid, potential)
    with pytest.raises(ValueError, match="grid shape"):
        tdgl_rhs(np.zeros(7), grid, potential, gamma=1.0)


def test_nonzero_neumann_boundary_is_consumed_by_existing_stencil():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.25,))
    x = grid.axes[0]
    gradient = 0.3
    phi = gradient * x
    potential = QuarticAgencityPotential(lambda_=0.0, mu=0.8)

    result = klein_gordon_acceleration(
        phi,
        grid,
        potential,
        boundary=NeumannBoundary(gradient=gradient),
    )
    expected = -potential.gradient(phi)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_small_amplitude_periodic_mode_matches_discrete_linear_dispersion():
    n = 64
    grid = _periodic_grid(n)
    x = grid.axes[0]
    mode = 3
    amplitude = 1.0e-4
    phi = amplitude * np.exp(1j * mode * x)
    potential = QuarticAgencityPotential(lambda_=-0.4, mu=1.3)

    acceleration = klein_gordon_acceleration(
        phi,
        grid,
        potential,
        boundary=PeriodicBoundary(),
    )
    dx = grid.spacings[0]
    discrete_k2 = 4.0 * np.sin(0.5 * mode * dx) ** 2 / dx**2
    exact_discrete = (
        potential.lambda_ - discrete_k2 - potential.mu * amplitude**2
    ) * phi
    linearized = (potential.lambda_ - discrete_k2) * phi

    np.testing.assert_allclose(acceleration, exact_discrete, rtol=1e-9, atol=1e-13)
    nonlinear_correction = np.linalg.norm(acceleration - linearized)
    linear_scale = np.linalg.norm(linearized)
    assert nonlinear_correction / linear_scale < 1e-7
