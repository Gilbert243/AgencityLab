"""Integration gates for source-defined 1.1.5 theory-completeness extensions.

These tests verify software transcription and cross-layer wiring only. They are
not a falsification protocol for the Theory of Agencity.
"""

from __future__ import annotations

import numpy as np

from agencitylab.extensions import (
    RIEMANNIAN_EXTENSION_STATUS,
    mean_contrast_criterion,
    orientational_entropy_criterion,
    riemannian_dynamic_intensity,
)
from agencitylab.fields import (
    FLAT_FIELD_METRIC_SIGNATURE,
    QuarticAgencityPotential,
    UniformRectilinearGrid,
    appendix_b_beta_equation_residual,
    dimensionless_static_residual,
    effective_beta_rhs,
    effective_beta_stationary_amplitude,
    phase_noether_current,
    u1_noether_current,
)
from agencitylab.fields.effective_beta import SCIENTIFIC_STATUS as EFFECTIVE_BETA_STATUS
from agencitylab.scientific_status import ScientificStatus


def test_effective_beta_stationary_branch_is_wired_without_touching_canonical_beta() -> None:
    grid = UniformRectilinearGrid(shape=(12,), spacings=(0.2,))
    amplitude = effective_beta_stationary_amplitude(
        linear_coefficient=1.2,
        saturation_coefficient=0.3,
    )
    beta = np.full(grid.shape, amplitude * np.exp(0.4j))
    rhs = effective_beta_rhs(
        beta,
        grid,
        diffusion_coefficient=0.5,
        linear_coefficient=1.2,
        saturation_coefficient=0.3,
        boundary="periodic",
    )
    np.testing.assert_allclose(rhs, 0.0, atol=1e-14)
    assert EFFECTIVE_BETA_STATUS is ScientificStatus.RESEARCH


def test_chapter16_noether_current_matches_phase_form() -> None:
    radius = np.array([1.4])
    theta_derivatives = np.array([[0.3, -0.2, 0.1, 0.05]])
    phi = radius * np.exp(0.7j)
    derivatives = 1j * phi[..., np.newaxis] * theta_derivatives
    np.testing.assert_allclose(
        u1_noether_current(phi, derivatives),
        phase_noether_current(radius, theta_derivatives),
        atol=1e-14,
        rtol=1e-14,
    )
    assert FLAT_FIELD_METRIC_SIGNATURE == (1, -1, -1, -1)


def test_appendix_b_equation_preserves_explicit_pc_squared_normalisation() -> None:
    potential = QuarticAgencityPotential(lambda_=1.0, mu=1.0)
    beta = np.array([0.2 + 0.1j])
    box_beta = np.array([0.05 - 0.03j])
    power = 2.0
    expected = box_beta + potential.gradient(beta) / power**2
    np.testing.assert_allclose(
        appendix_b_beta_equation_residual(
            beta,
            box_beta,
            potential,
            P_c=power,
        ),
        expected,
    )


def test_riemannian_definition_124_reduces_to_euclidean_formula() -> None:
    velocity = np.array([3.0, 4.0])
    acceleration = np.array([2.0, -1.0])
    metric = np.eye(2)
    projection = acceleration @ velocity
    expected = np.sqrt(velocity @ velocity + projection**2)
    assert riemannian_dynamic_intensity(velocity, acceleration, metric) == expected
    assert RIEMANNIAN_EXTENSION_STATUS is ScientificStatus.EXPERIMENTAL


def test_dimensionless_vacuum_and_window_criteria_are_available() -> None:
    grid = UniformRectilinearGrid(shape=(8,), spacings=(0.25,))
    psi = np.full(grid.shape, np.exp(0.2j))
    np.testing.assert_allclose(
        dimensionless_static_residual(psi, grid, boundary="periodic"),
        0.0,
        atol=1e-14,
    )

    assert mean_contrast_criterion(np.array([-1.0, 2.0, -3.0])) == 2.0
    entropy = orientational_entropy_criterion(
        np.array([-0.75, -0.25, 0.25, 0.75]),
        bin_edges=np.array([-1.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(entropy, np.log(2.0))
