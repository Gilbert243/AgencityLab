"""Cross-layer release checks for AgencityLab 1.1.3 research extensions."""

from __future__ import annotations

import numpy as np

import agencitylab
from agencitylab.gravity import minkowski_box


def test_dissipative_field_solution_feeds_thermodynamic_dissipation():
    grid = agencitylab.UniformRectilinearGrid(shape=(16,), spacings=(0.5,))
    potential = agencitylab.QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    gamma = 0.25
    solution = agencitylab.simulate_dissipative_klein_gordon(
        np.full(grid.shape, 0.4, dtype=float),
        np.zeros(grid.shape, dtype=float),
        grid,
        potential,
        gamma=gamma,
        dt=0.01,
        n_steps=2,
    )

    assert solution.phi_dot is not None
    density = agencitylab.dissipation_density(solution.phi_dot[1], gamma)
    np.testing.assert_allclose(density, gamma * np.abs(solution.phi_dot[1]) ** 2)
    assert solution.parameters["Gamma"] == gamma
    assert "Gamma" in solution.parameter_provenance


def test_stationary_vacuum_has_zero_dissipation_and_entropy_production():
    grid = agencitylab.UniformRectilinearGrid(shape=(8,), spacings=(0.5,))
    vacuum = agencitylab.vacuum_state(1.0, 1.0, theta=0.37)
    phi_dot = np.zeros(grid.shape, dtype=complex)

    np.testing.assert_array_equal(
        agencitylab.dissipation_density(phi_dot, gamma=2.0),
        0.0,
    )
    np.testing.assert_array_equal(
        agencitylab.entropy_production_density(phi_dot, gamma=2.0, t_eff=3.0),
        0.0,
    )

    potential = agencitylab.QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    phi = np.full(grid.shape, vacuum, dtype=complex)
    np.testing.assert_allclose(
        agencitylab.klein_gordon_acceleration(phi, grid, potential),
        0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        agencitylab.curved_field_residual(
            np.zeros_like(phi),
            phi,
            potential,
            0.0,
            xi=0.0,
        ),
        0.0,
        atol=1e-14,
    )


def test_gravity_reuses_shared_quartic_potential_without_duplicate_force():
    potential = agencitylab.QuarticAgencityPotential(lambda_=1.3, mu=0.7)
    phi = np.array([0.2 + 0.1j, -0.4 + 0.3j, 0.0j])
    force = potential.gradient(phi)

    residual = agencitylab.curved_field_residual(
        -force,
        phi,
        potential,
        0.0,
        xi=0.0,
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-15)


def test_metric_signature_difference_is_explicit_not_silently_reconciled():
    assert agencitylab.GRAVITY_METRIC_SIGNATURE == (-1, 1, 1, 1)
    metric = agencitylab.minkowski_metric()
    np.testing.assert_array_equal(np.diag(metric), np.array([-1.0, 1.0, 1.0, 1.0]))

    phi_tt = np.array([1.5, -0.25])
    spatial_laplacian = np.array([0.5, 2.0])
    np.testing.assert_array_equal(
        minkowski_box(phi_tt, spatial_laplacian),
        -phi_tt + spatial_laplacian,
    )


def test_field_entropy_is_global_u1_invariant():
    grid = agencitylab.UniformRectilinearGrid(shape=(12,), spacings=(0.25,))
    x = grid.axes[0]
    phi = 0.4 * np.exp(1j * x)
    rotated = phi * np.exp(1j * 0.73)

    original = agencitylab.field_agencial_entropy(phi, 2.0, grid)
    transformed = agencitylab.field_agencial_entropy(rotated, 2.0, grid)
    np.testing.assert_allclose(transformed, original, rtol=1e-14, atol=1e-14)


def test_v113_selected_research_apis_are_public_without_redefining_scalar_core():
    assert agencitylab.__version__ == "1.1.4"
    assert callable(agencitylab.temperature_dependent_lambda)
    assert callable(agencitylab.modulus_law_margin)
    assert callable(agencitylab.phase_law_prediction)
    assert callable(agencitylab.stress_energy_tensor)
    assert callable(agencitylab.einstein_equation_residual)

    xi = np.arange(32.0)
    result = agencitylab.compute_agencity(
        u=np.sin(0.2 * xi),
        xi=xi,
        A_ref=1.0,
        tau=2.0,
        w=2.0,
        P_c=3.0,
    )
    np.testing.assert_allclose(result.b, result.P_c * result.beta)
