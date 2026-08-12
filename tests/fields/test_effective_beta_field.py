from __future__ import annotations

import numpy as np
import pytest

from agencitylab.fields.effective_beta import (
    SCIENTIFIC_STATUS,
    effective_beta_reaction,
    effective_beta_rhs,
    effective_beta_stationary_amplitude,
)
from agencitylab.fields.numerics import UniformRectilinearGrid, gradient, laplacian
from agencitylab.scientific_status import ScientificStatus


def _grid(n: int = 32) -> UniformRectilinearGrid:
    return UniformRectilinearGrid(shape=(n,), spacings=(2.0 * np.pi / n,))


def test_effective_beta_layer_is_research() -> None:
    assert SCIENTIFIC_STATUS is ScientificStatus.RESEARCH


def test_reaction_zero_and_nonzero_stationary_branches() -> None:
    assert effective_beta_reaction(
        np.array([0.0 + 0.0j]),
        linear_coefficient=2.0,
        saturation_coefficient=0.5,
    )[0] == 0.0

    amplitude = effective_beta_stationary_amplitude(
        linear_coefficient=2.0,
        saturation_coefficient=0.5,
    )
    beta = amplitude * np.exp(0.73j)
    reaction = effective_beta_reaction(
        np.array([beta]),
        linear_coefficient=2.0,
        saturation_coefficient=0.5,
    )
    np.testing.assert_allclose(reaction, 0.0, atol=1e-14)


def test_stationary_amplitude_requires_positive_source_branch() -> None:
    with pytest.raises(ValueError):
        effective_beta_stationary_amplitude(
            linear_coefficient=0.0,
            saturation_coefficient=1.0,
        )
    with pytest.raises(ValueError):
        effective_beta_stationary_amplitude(
            linear_coefficient=1.0,
            saturation_coefficient=0.0,
        )


def test_rhs_matches_direct_source_decomposition() -> None:
    grid = _grid()
    x = grid.axes[0]
    beta = np.exp(2.0j * x)
    velocity = (0.4,)
    diffusion = 0.7
    linear = -0.2
    saturation = 0.3

    result = effective_beta_rhs(
        beta,
        grid,
        diffusion_coefficient=diffusion,
        linear_coefficient=linear,
        saturation_coefficient=saturation,
        velocity=velocity,
        boundary="periodic",
    )
    expected = (
        diffusion * laplacian(beta, grid, boundary="periodic")
        + effective_beta_reaction(
            beta,
            linear_coefficient=linear,
            saturation_coefficient=saturation,
        )
        - velocity[0] * gradient(beta, grid, boundary="periodic")[0]
    )
    np.testing.assert_allclose(result, expected)


def test_rhs_is_covariant_under_global_u1_phase() -> None:
    grid = _grid()
    x = grid.axes[0]
    beta = 0.6 * np.exp(1.7j * x)
    phase = np.exp(0.41j)
    kwargs = dict(
        diffusion_coefficient=0.9,
        linear_coefficient=0.3,
        saturation_coefficient=0.8,
        velocity=(0.2,),
        boundary="periodic",
    )
    original = effective_beta_rhs(beta, grid, **kwargs)
    rotated = effective_beta_rhs(phase * beta, grid, **kwargs)
    np.testing.assert_allclose(rotated, phase * original, atol=1e-13, rtol=1e-13)


def test_homogeneous_stationary_pattern_has_zero_rhs() -> None:
    grid = _grid(8)
    amplitude = effective_beta_stationary_amplitude(
        linear_coefficient=1.2,
        saturation_coefficient=0.3,
    )
    beta = np.full(grid.shape, amplitude * np.exp(-0.2j))
    rhs = effective_beta_rhs(
        beta,
        grid,
        diffusion_coefficient=0.5,
        linear_coefficient=1.2,
        saturation_coefficient=0.3,
        boundary="periodic",
    )
    np.testing.assert_allclose(rhs, 0.0, atol=1e-14)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_diffusion_coefficient_must_be_positive_and_finite(bad: float) -> None:
    grid = _grid(8)
    with pytest.raises(ValueError):
        effective_beta_rhs(
            np.zeros(grid.shape),
            grid,
            diffusion_coefficient=bad,
            linear_coefficient=0.0,
            saturation_coefficient=1.0,
        )


def test_velocity_shape_and_reaction_validation() -> None:
    grid = _grid(8)
    with pytest.raises(ValueError):
        effective_beta_rhs(
            np.zeros(grid.shape),
            grid,
            diffusion_coefficient=1.0,
            linear_coefficient=0.0,
            saturation_coefficient=1.0,
            velocity=(1.0, 2.0),
        )
    with pytest.raises(ValueError):
        effective_beta_reaction(
            np.ones(3),
            linear_coefficient=0.0,
            saturation_coefficient=-1.0,
        )
