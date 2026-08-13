from __future__ import annotations

import numpy as np
import pytest

from agencitylab.fields.dynamics import (
    simulate_dissipative_klein_gordon,
    simulate_klein_gordon,
    simulate_tdgl,
)
from agencitylab.fields.numerics import (
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
    UniformRectilinearGrid,
    gradient_norm_squared,
)
from agencitylab.fields.physics import (
    QuarticAgencityPotential,
    field_energy_density,
    total_field_energy,
)
from agencitylab.models.field_extensions import (
    DynamicalAgencityFieldSolution,
    ParameterSource,
)
from agencitylab.scientific_status import ScientificStatus


def _periodic_grid(n: int = 32) -> UniformRectilinearGrid:
    return UniformRectilinearGrid(shape=(n,), spacings=(2.0 * np.pi / n,))


def _energy_at(
    solution: DynamicalAgencityFieldSolution,
    index: int,
    grid: UniformRectilinearGrid,
    potential: QuarticAgencityPotential,
) -> float:
    assert solution.phi_dot is not None
    phi = solution.phi[index]
    phi_dot = solution.phi_dot[index]
    grad_sq = gradient_norm_squared(phi, grid, boundary=PeriodicBoundary())
    density = field_energy_density(phi, phi_dot, grad_sq, potential)
    return total_field_energy(density, volume_element=grid.cell_volume)


def test_simulators_return_shared_solution_with_metadata_and_provenance():
    grid = _periodic_grid(16)
    x = grid.axes[0]
    phi0 = 0.1 * np.exp(1j * x)
    phi_dot0 = 0.02j * phi0
    potential = QuarticAgencityPotential(lambda_=-0.5, mu=1.0)

    conservative = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.01,
        n_steps=3,
    )
    dissipative = simulate_dissipative_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        gamma=0.3,
        dt=0.01,
        n_steps=3,
    )
    overdamped = simulate_tdgl(
        phi0,
        grid,
        potential,
        gamma=0.8,
        dt=0.01,
        n_steps=3,
    )

    for solution in (conservative, dissipative, overdamped):
        assert isinstance(solution, DynamicalAgencityFieldSolution)
        assert solution.scientific_status is ScientificStatus.RESEARCH
        assert solution.phi.shape == (4, 16)
        assert solution.spatial_shape == grid.shape
        assert solution.boundary_name == "periodic"
        assert solution.parameters["lambda"] == potential.lambda_
        assert solution.parameters["mu"] == potential.mu
        assert solution.parameters["dt"] == 0.01
        assert solution.parameter_provenance["dt"].source is ParameterSource.USER_SUPPLIED
        assert solution.solver_metadata["fixed_step"] is True
        assert solution.units_convention == potential.units_convention

    assert conservative.phi_dot is not None
    assert conservative.solver_metadata["integrator"] == "velocity_verlet"
    assert dissipative.phi_dot is not None
    assert dissipative.solver_metadata["integrator"] == "rk4"
    assert dissipative.parameters["Gamma"] == 0.3
    assert overdamped.phi_dot is None
    assert overdamped.solver_metadata["integrator"] == "rk4"
    assert overdamped.parameters["Gamma"] == 0.8


def test_dirichlet_state_projection_is_explicit_and_preserved_by_simulators():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)
    boundary = DirichletBoundary(value=0.25)
    phi0 = np.linspace(-0.4, 0.5, 8)
    phi_dot0 = np.linspace(0.3, -0.2, 8)
    phi0_before = phi0.copy()
    phi_dot0_before = phi_dot0.copy()

    conservative = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.002,
        n_steps=4,
        boundary=boundary,
    )
    dissipative = simulate_dissipative_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        gamma=0.4,
        dt=0.002,
        n_steps=4,
        boundary=boundary,
    )
    overdamped = simulate_tdgl(
        phi0,
        grid,
        potential,
        gamma=0.7,
        dt=0.002,
        n_steps=4,
        boundary=boundary,
    )

    np.testing.assert_array_equal(phi0, phi0_before)
    np.testing.assert_array_equal(phi_dot0, phi_dot0_before)
    for solution in (conservative, dissipative, overdamped):
        np.testing.assert_array_equal(solution.phi[:, 0], 0.25)
        np.testing.assert_array_equal(solution.phi[:, -1], 0.25)
        assert solution.boundary_name == "dirichlet"
        assert solution.solver_metadata["state_boundary_projection"] is True
    for solution in (conservative, dissipative):
        assert solution.phi_dot is not None
        np.testing.assert_array_equal(solution.phi_dot[:, 0], 0.0)
        np.testing.assert_array_equal(solution.phi_dot[:, -1], 0.0)


def test_neumann_simulation_reports_stencil_only_boundary_handling():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    x = grid.axes[0]
    phi0 = 0.1 * x
    potential = QuarticAgencityPotential(lambda_=-0.5, mu=1.0)
    solution = simulate_tdgl(
        phi0,
        grid,
        potential,
        gamma=1.0,
        dt=0.001,
        n_steps=2,
        boundary=NeumannBoundary(gradient=0.1),
    )
    assert solution.boundary_name == "neumann"
    assert solution.solver_metadata["state_boundary_projection"] is False
    assert solution.solver_metadata["boundary_handling"] == (
        "stencil_only_no_state_projection"
    )


@pytest.mark.parametrize(
    ("dt", "n_steps"),
    [
        (0.0, 1),
        (-0.1, 1),
        (np.nan, 1),
        (np.inf, 1),
        (0.1, 0),
        (0.1, -1),
        (0.1, 1.5),
        (0.1, True),
    ],
)
def test_simulators_reject_invalid_fixed_time_controls(dt, n_steps):
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    potential = QuarticAgencityPotential(lambda_=0.0, mu=1.0)
    with pytest.raises(ValueError):
        simulate_tdgl(
            np.zeros(grid.shape),
            grid,
            potential,
            gamma=1.0,
            dt=dt,
            n_steps=n_steps,
        )


def test_tdgl_rk4_shows_fourth_order_temporal_convergence_on_homogeneous_case():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi0_value = 0.8
    phi0 = np.full(grid.shape, phi0_value)
    mu = 1.5
    gamma = 1.0
    potential = QuarticAgencityPotential(lambda_=0.0, mu=mu)
    final_time = 0.4

    coarse = simulate_tdgl(
        phi0,
        grid,
        potential,
        gamma=gamma,
        dt=0.2,
        n_steps=2,
    )
    fine = simulate_tdgl(
        phi0,
        grid,
        potential,
        gamma=gamma,
        dt=0.1,
        n_steps=4,
    )
    exact = phi0_value / np.sqrt(
        1.0 + 2.0 * mu * phi0_value**2 * final_time / gamma
    )
    coarse_error = abs(coarse.phi[-1, 0] - exact)
    fine_error = abs(fine.phi[-1, 0] - exact)

    assert coarse_error > fine_error > 0.0
    assert coarse_error / fine_error > 8.0


def test_velocity_verlet_shows_second_order_temporal_convergence():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi0 = np.full(grid.shape, 0.3)
    phi_dot0 = np.full(grid.shape, 0.2)
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=0.5)

    coarse = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.04,
        n_steps=20,
    )
    fine = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.02,
        n_steps=40,
    )
    reference = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.002,
        n_steps=400,
    )
    reference_value = reference.phi[-1, 0]
    coarse_error = abs(coarse.phi[-1, 0] - reference_value)
    fine_error = abs(fine.phi[-1, 0] - reference_value)

    assert coarse_error > fine_error > 0.0
    assert coarse_error / fine_error > 3.0


def test_conservative_energy_is_numerically_nearly_conserved():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi0 = np.full(grid.shape, 0.3)
    phi_dot0 = np.full(grid.shape, 0.2)
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=0.5)

    solution = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.002,
        n_steps=200,
        boundary=PeriodicBoundary(),
    )
    initial = _energy_at(solution, 0, grid, potential)
    final = _energy_at(solution, -1, grid, potential)
    relative_drift = abs(final - initial) / initial

    # Numerical diagnostic only; this tolerance is not a physical law.
    assert relative_drift < 2e-4


def test_dissipative_energy_decreases_on_homogeneous_controlled_case():
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.2,))
    phi0 = np.full(grid.shape, 0.2)
    phi_dot0 = np.full(grid.shape, 0.3)
    potential = QuarticAgencityPotential(lambda_=-1.0, mu=1.0)

    solution = simulate_dissipative_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        gamma=0.5,
        dt=0.005,
        n_steps=100,
        boundary=PeriodicBoundary(),
    )
    initial = _energy_at(solution, 0, grid, potential)
    final = _energy_at(solution, -1, grid, potential)

    assert final < initial
