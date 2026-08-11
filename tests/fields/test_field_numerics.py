"""Analytic and convergence tests for generic field numerical infrastructure."""

from __future__ import annotations

import numpy as np
import pytest

from agencitylab.fields.numerics import (
    DirichletBoundary,
    NeumannBoundary,
    PeriodicBoundary,
    UniformRectilinearGrid,
    diffusion_cfl_limit,
    gradient,
    gradient_norm_squared,
    integrate_spatial,
    laplacian,
    rk4_step,
    velocity_verlet_step,
    wave_cfl_limit,
)


def periodic_grid_1d(n: int, length: float = 2.0 * np.pi) -> UniformRectilinearGrid:
    x = np.linspace(0.0, length, n, endpoint=False)
    return UniformRectilinearGrid([x])


def test_grid_from_explicit_axes_1d_2d_3d_and_nd():
    for ndim in (1, 2, 3, 5):
        axes = [np.linspace(float(i), float(i) + 1.0, 5 + i) for i in range(ndim)]
        grid = UniformRectilinearGrid(axes)
        assert grid.ndim == ndim
        assert grid.shape == tuple(axis.size for axis in axes)
        assert len(grid.spacings) == ndim
        assert grid.cell_volume == pytest.approx(np.prod(grid.spacings))
        assert grid.volume_element == grid.cell_volume
        assert grid.extent == tuple((axis[0], axis[-1]) for axis in axes)


def test_grid_from_shape_spacings_and_origins():
    grid = UniformRectilinearGrid(shape=(4, 5, 6), spacings=(0.5, 0.25, 2.0), origins=(-1.0, 3.0, 4.0))
    assert grid.shape == (4, 5, 6)
    assert grid.spacings == pytest.approx((0.5, 0.25, 2.0))
    assert grid.origins == pytest.approx((-1.0, 3.0, 4.0))
    np.testing.assert_allclose(grid.axes[0], [-1.0, -0.5, 0.0, 0.5])


@pytest.mark.parametrize(
    "axes",
    [
        [np.array([0.0, 0.5, 1.1])],
        [np.array([0.0, 1.0, 0.5])],
        [np.array([0.0, np.nan, 1.0])],
        [np.array([[0.0, 1.0], [2.0, 3.0]])],
    ],
)
def test_invalid_explicit_axes_rejected(axes):
    with pytest.raises(ValueError):
        UniformRectilinearGrid(axes)


def test_invalid_generated_grid_rejected():
    with pytest.raises(ValueError):
        UniformRectilinearGrid(shape=(4, 5), spacings=(1.0,))
    with pytest.raises(ValueError):
        UniformRectilinearGrid(shape=(4,), spacings=(0.0,))
    with pytest.raises(ValueError):
        UniformRectilinearGrid(shape=(4,), spacings=(-1.0,))
    with pytest.raises(ValueError):
        UniformRectilinearGrid(shape=(1,), spacings=(1.0,))


def test_periodic_gradient_sine_and_second_order_convergence():
    errors = []
    for n in (64, 128):
        grid = periodic_grid_1d(n)
        x = grid.axes[0]
        k = 3.0
        numerical = gradient(np.sin(k * x), grid, PeriodicBoundary())[0]
        exact = k * np.cos(k * x)
        errors.append(np.max(np.abs(numerical - exact)))
    assert errors[0] / errors[1] > 3.7


def test_periodic_laplacian_sine_1d_and_2d():
    grid1 = periodic_grid_1d(128)
    x = grid1.axes[0]
    k = 2.0
    field1 = np.sin(k * x)
    exact1 = -(k**2) * field1
    np.testing.assert_allclose(laplacian(field1, grid1), exact1, atol=8e-3, rtol=8e-3)

    n = 96
    axis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    grid2 = UniformRectilinearGrid([axis, axis])
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    field2 = np.sin(2.0 * xx) + np.sin(3.0 * yy)
    exact2 = -4.0 * np.sin(2.0 * xx) - 9.0 * np.sin(3.0 * yy)
    np.testing.assert_allclose(laplacian(field2, grid2), exact2, atol=5e-2, rtol=5e-2)


def test_periodic_complex_plane_wave_preserves_complex_dtype():
    grid = periodic_grid_1d(256)
    x = grid.axes[0]
    k = 4.0
    phi = np.exp(1j * k * x)
    grad = gradient(phi, grid)[0]
    lap = laplacian(phi, grid)
    assert np.iscomplexobj(grad)
    assert np.iscomplexobj(lap)
    np.testing.assert_allclose(grad, 1j * k * phi, atol=7e-3, rtol=7e-3)
    np.testing.assert_allclose(lap, -(k**2) * phi, atol=4e-2, rtol=4e-2)


def test_constant_field_has_zero_periodic_and_homogeneous_neumann_operators():
    grid = UniformRectilinearGrid(shape=(7, 8, 9), spacings=(0.2, 0.3, 0.4))
    field = np.full(grid.shape, 2.5 - 1.2j)
    for boundary in (PeriodicBoundary(), NeumannBoundary(0.0)):
        for component in gradient(field, grid, boundary):
            np.testing.assert_allclose(component, 0.0, atol=1e-14)
        np.testing.assert_allclose(laplacian(field, grid, boundary), 0.0, atol=1e-14)


def test_dirichlet_boundary_profile():
    x = np.linspace(0.0, 1.0, 65)
    grid = UniformRectilinearGrid([x])
    field = x * (1.0 - x)
    bc = DirichletBoundary(0.0)
    grad = gradient(field, grid, bc)[0]
    lap = laplacian(field, grid, bc)
    np.testing.assert_allclose(grad, 1.0 - 2.0 * x, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(lap, -2.0, atol=2e-10, rtol=2e-10)


def test_dirichlet_value_is_imposed_before_differentiation():
    x = np.linspace(0.0, 1.0, 9)
    grid = UniformRectilinearGrid([x])
    field = x * (1.0 - x)
    field[[0, -1]] = 99.0
    lap = laplacian(field, grid, DirichletBoundary(0.0))
    np.testing.assert_allclose(lap, -2.0, atol=1e-12)


def test_neumann_linear_profile_has_prescribed_derivative_and_zero_laplacian():
    x = np.linspace(-2.0, 3.0, 81)
    grid = UniformRectilinearGrid([x])
    field = 1.7 * x - 0.4
    bc = NeumannBoundary(gradient=1.7)
    np.testing.assert_allclose(gradient(field, grid, bc)[0], 1.7, atol=2e-13)
    np.testing.assert_allclose(laplacian(field, grid, bc), 0.0, atol=2e-12)


def test_gradient_norm_squared_is_real_nonnegative_and_matches_components():
    n = 64
    axis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    grid = UniformRectilinearGrid([axis, axis])
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    field = np.exp(1j * xx) + 0.5j * np.exp(2j * yy)
    components = gradient(field, grid)
    norm2 = gradient_norm_squared(field, grid)
    expected = sum(np.abs(component) ** 2 for component in components)
    assert not np.iscomplexobj(norm2)
    assert np.min(norm2) >= 0.0
    np.testing.assert_allclose(norm2, expected)


def test_integrate_spatial_rectangular_rule_1d_2d_3d_and_complex():
    grids = [
        UniformRectilinearGrid(shape=(5,), spacings=(0.2,)),
        UniformRectilinearGrid(shape=(4, 6), spacings=(0.5, 0.25)),
        UniformRectilinearGrid(shape=(3, 4, 5), spacings=(0.2, 0.3, 0.4)),
    ]
    for grid in grids:
        density = np.full(grid.shape, 3.0)
        expected = density.size * 3.0 * grid.cell_volume
        assert integrate_spatial(density, grid) == pytest.approx(expected)

    grid = UniformRectilinearGrid(shape=(7,), spacings=(0.1,))
    density = np.arange(7) * (1.0 + 2.0j)
    assert integrate_spatial(density, grid) == pytest.approx(np.sum(density) * 0.1)


def test_integrate_spatial_simple_function_uses_documented_rectangle_rule():
    x = np.linspace(0.0, 1.0, 11)
    grid = UniformRectilinearGrid([x])
    assert integrate_spatial(x**2, grid) == pytest.approx(np.sum(x**2) * grid.spacings[0])


def _integrate_rk4(dt: float) -> complex:
    omega = 2.0
    y = np.array([1.0 + 0.0j])
    t = 0.0
    steps = round(1.0 / dt)
    for _ in range(steps):
        y = rk4_step(lambda time, state: 1j * omega * state, t, y, dt)
        t += dt
    return y[0]


def test_rk4_complex_oscillator_and_fourth_order_convergence():
    exact = np.exp(2.0j)
    coarse = abs(_integrate_rk4(0.1) - exact)
    fine = abs(_integrate_rk4(0.05) - exact)
    assert coarse / fine > 14.0


def _integrate_verlet(dt: float) -> tuple[float, float, float]:
    omega = 1.3
    q = np.array([1.0])
    v = np.array([0.0])
    t = 0.0
    initial_energy = 0.5 * (v[0] ** 2 + omega**2 * q[0] ** 2)
    max_energy_error = 0.0
    steps = round(4.0 / dt)
    for _ in range(steps):
        q, v = velocity_verlet_step(
            lambda time, position, velocity: -(omega**2) * position,
            t,
            q,
            v,
            dt,
        )
        t += dt
        energy = 0.5 * (v[0] ** 2 + omega**2 * q[0] ** 2)
        max_energy_error = max(max_energy_error, abs(energy - initial_energy))
    exact_q = np.cos(omega * t)
    return abs(q[0] - exact_q), max_energy_error, initial_energy


def test_velocity_verlet_harmonic_oscillator_second_order_and_energy_behavior():
    coarse_error, coarse_drift, energy = _integrate_verlet(0.04)
    fine_error, fine_drift, _ = _integrate_verlet(0.02)
    assert coarse_error / fine_error > 3.7
    assert fine_drift < coarse_drift
    assert fine_drift / energy < 2e-3


def test_velocity_verlet_supports_complex_arrays():
    q = np.array([1.0 + 1.0j, 0.5 - 0.25j])
    v = np.array([0.0j, 0.0j])
    q_new, v_new = velocity_verlet_step(
        lambda t, position, velocity: -position,
        0.0,
        q,
        v,
        0.01,
    )
    assert q_new.shape == q.shape
    assert v_new.shape == v.shape
    assert np.iscomplexobj(q_new)
    assert np.iscomplexobj(v_new)


def test_cfl_helpers_match_standard_sufficient_guidelines():
    grid = UniformRectilinearGrid(shape=(8, 9), spacings=(0.5, 0.25))
    inverse_squared_sum = 1.0 / 0.5**2 + 1.0 / 0.25**2
    assert wave_cfl_limit(grid, 2.0) == pytest.approx(1.0 / (2.0 * np.sqrt(inverse_squared_sum)))
    assert diffusion_cfl_limit(grid, 0.3) == pytest.approx(1.0 / (2.0 * 0.3 * inverse_squared_sum))
    assert np.isinf(wave_cfl_limit(grid, 0.0))
    assert np.isinf(diffusion_cfl_limit(grid, 0.0))


def test_small_grids_are_rejected_explicitly_for_stencils():
    periodic = UniformRectilinearGrid(shape=(2,), spacings=(1.0,))
    with pytest.raises(ValueError, match="at least 3"):
        gradient(np.zeros(2), periodic, PeriodicBoundary())
    dirichlet = UniformRectilinearGrid(shape=(3,), spacings=(1.0,))
    with pytest.raises(ValueError, match="at least 4"):
        laplacian(np.zeros(3), dirichlet, DirichletBoundary())


def test_validation_rejects_bad_fields_boundaries_steps_and_callbacks():
    grid = UniformRectilinearGrid(shape=(5,), spacings=(1.0,))
    with pytest.raises(ValueError):
        gradient(np.zeros(4), grid)
    with pytest.raises(ValueError):
        gradient(np.array([0.0, 1.0, np.nan, 3.0, 4.0]), grid)
    with pytest.raises(ValueError):
        gradient(np.zeros(5), grid, "unknown")
    with pytest.raises(ValueError):
        rk4_step(lambda t, y: y, 0.0, np.ones(2), 0.0)
    with pytest.raises(ValueError, match="returned shape"):
        rk4_step(lambda t, y: np.ones(3), 0.0, np.ones(2), 0.1)
    with pytest.raises(ValueError, match="returned shape"):
        velocity_verlet_step(
            lambda t, q, v: np.ones(3),
            0.0,
            np.ones(2),
            np.zeros(2),
            0.1,
        )
    with pytest.raises(ValueError):
        wave_cfl_limit(grid, -1.0)
    with pytest.raises(ValueError):
        diffusion_cfl_limit(grid, np.nan)
