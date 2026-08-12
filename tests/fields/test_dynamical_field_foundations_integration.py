import numpy as np

import agencitylab
from agencitylab.fields import (
    PeriodicBoundary,
    QuarticAgencityPotential,
    UniformRectilinearGrid,
    beta_to_phi,
    field_energy_density,
    gradient_norm_squared,
    laplacian,
)
from agencitylab.models import DynamicalAgencityFieldState
from agencitylab.scientific_status import ScientificStatus


def test_public_foundations_are_exposed_with_expected_statuses():
    assert agencitylab.__version__ == "1.1.6"
    assert agencitylab.ScientificStatus.RESEARCH is ScientificStatus.RESEARCH
    assert agencitylab.DynamicalAgencityFieldState is DynamicalAgencityFieldState
    assert agencitylab.QuarticAgencityPotential is QuarticAgencityPotential
    assert agencitylab.UniformRectilinearGrid is UniformRectilinearGrid


def test_observable_beta_bridge_initializes_research_state_without_mutation():
    beta = np.array([[1.0 + 1.0j, 2.0], [3.0, 4.0 - 1.0j]])
    beta_before = beta.copy()
    p_c = np.array([0.0, 2.0])
    tau = np.array([1.5, 0.5])

    phi = beta_to_phi(beta, p_c, tau, time_axis=0)

    expected = np.sqrt(p_c[None, :] * tau[None, :]) * beta
    np.testing.assert_allclose(phi, expected)
    np.testing.assert_array_equal(beta, beta_before)
    np.testing.assert_array_equal(phi[:, 0], 0.0)

    state = DynamicalAgencityFieldState(
        phi=phi[-1],
        phi_dot=np.zeros_like(phi[-1]),
        time=1.0,
        spatial_shape=(2,),
        scientific_status=ScientificStatus.RESEARCH,
        units_convention="dimensionless",
    )
    np.testing.assert_allclose(state.phi, phi[-1])
    assert state.scientific_status is ScientificStatus.RESEARCH


def test_physics_and_numerics_share_one_field_without_formula_duplication():
    n = 64
    dx = 2.0 * np.pi / n
    grid = UniformRectilinearGrid(shape=(n,), spacings=(dx,))
    x = grid.axes[0]
    phi = np.exp(1j * x)
    phi_dot = 0.25j * phi
    boundary = PeriodicBoundary()

    grad_sq = gradient_norm_squared(phi, grid, boundary=boundary)
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    density = field_energy_density(phi, phi_dot, grad_sq, potential)

    assert density.shape == grid.shape
    assert np.isrealobj(density)
    assert np.all(np.isfinite(density))

    numerical_laplacian = laplacian(phi, grid, boundary=boundary)
    np.testing.assert_allclose(numerical_laplacian, -phi, rtol=5e-3, atol=5e-3)


def test_status_boundary_remains_explicit():
    assert agencitylab.compute_agencity_field is not None
    assert ScientificStatus.EXPERIMENTAL.value == "experimental"
    assert ScientificStatus.RESEARCH.value == "research"
