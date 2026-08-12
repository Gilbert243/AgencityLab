from __future__ import annotations

import numpy as np
import pytest

import agencitylab
import agencitylab.fields as fields_api
from agencitylab.fields.dynamics import (
    SCIENTIFIC_STATUS,
    dissipative_klein_gordon_acceleration,
    klein_gordon_acceleration,
    simulate_dissipative_klein_gordon,
    simulate_klein_gordon,
    simulate_tdgl,
    tdgl_rhs,
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


def test_dynamics_status_and_export_boundary_are_research_only():
    assert SCIENTIFIC_STATUS is ScientificStatus.RESEARCH
    assert agencitylab.__version__ == "1.1.1"
    assert not hasattr(fields_api, "simulate_klein_gordon")
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
