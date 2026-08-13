"""Cross-layer integration checks for research field dynamics and coherent structures."""

from __future__ import annotations

import numpy as np

import agencitylab
import agencitylab.fields as fields_api
from agencitylab.fields import (
    NeumannBoundary,
    PeriodicBoundary,
    QuarticAgencityPotential,
    UniformRectilinearGrid,
    domain_wall_profile,
    domain_wall_residual,
    klein_gordon_acceleration,
    phase_winding,
    simulate_dissipative_klein_gordon,
    simulate_klein_gordon,
    simulate_tdgl,
    vacuum_state,
)


def test_public_api_exposes_dynamics_and_coherent_references_in_fields_namespace():
    names = (
        "klein_gordon_acceleration",
        "dissipative_klein_gordon_acceleration",
        "tdgl_rhs",
        "simulate_klein_gordon",
        "simulate_dissipative_klein_gordon",
        "simulate_tdgl",
        "domain_wall_profile",
        "domain_wall_residual",
        "vortex_field",
        "vortex_radial_residual",
        "phase_winding",
        "field_zero_mask",
    )
    for name in names:
        assert hasattr(fields_api, name), name
        assert not hasattr(agencitylab, name), name


def test_domain_wall_static_residual_is_the_kg_acceleration():
    x = np.linspace(-8.0, 8.0, 257)
    grid = UniformRectilinearGrid(axes=(x,))
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    wall = domain_wall_profile(x, lambda_=1.0, mu=1.0)
    boundary = NeumannBoundary(gradient=0.0)

    residual = domain_wall_residual(
        wall,
        grid,
        lambda_=1.0,
        mu=1.0,
        boundary=boundary,
    )
    acceleration = klein_gordon_acceleration(
        wall,
        grid,
        potential,
        boundary=boundary,
    )

    np.testing.assert_allclose(acceleration, residual, rtol=0.0, atol=0.0)


def test_uniform_broken_vacuum_remains_stationary_in_all_three_dynamics():
    grid = UniformRectilinearGrid(shape=(32,), spacings=(0.25,))
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    vacuum = vacuum_state(1.0, 1.0, theta=0.37)
    phi0 = np.full(grid.shape, vacuum, dtype=complex)
    phi_dot0 = np.zeros(grid.shape, dtype=complex)
    boundary = PeriodicBoundary()

    conservative = simulate_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        dt=0.01,
        n_steps=3,
        boundary=boundary,
    )
    dissipative = simulate_dissipative_klein_gordon(
        phi0,
        phi_dot0,
        grid,
        potential,
        gamma=0.4,
        dt=0.01,
        n_steps=3,
        boundary=boundary,
    )
    overdamped = simulate_tdgl(
        phi0,
        grid,
        potential,
        gamma=1.0,
        dt=0.01,
        n_steps=3,
        boundary=boundary,
    )

    expected = np.broadcast_to(phi0, conservative.phi.shape)
    np.testing.assert_allclose(conservative.phi, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(dissipative.phi, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(overdamped.phi, expected, rtol=0.0, atol=1e-12)
    assert conservative.scientific_status is agencitylab.ScientificStatus.RESEARCH
    assert dissipative.scientific_status is agencitylab.ScientificStatus.RESEARCH
    assert overdamped.scientific_status is agencitylab.ScientificStatus.RESEARCH


def test_spatial_phase_winding_is_independent_of_global_u1_phase():
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    contour = np.exp(2.0j * theta)
    rotated = contour * np.exp(0.731j)

    assert phase_winding(contour) == 2.0
    assert phase_winding(rotated) == 2.0
