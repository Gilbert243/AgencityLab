"""Tests for research coherent structures and autonomous-field topology."""

import numpy as np
import pytest

from agencitylab.fields.coherent import (
    SCIENTIFIC_STATUS,
    domain_wall_profile,
    domain_wall_residual,
    field_zero_mask,
    phase_winding,
    vortex_field,
    vortex_radial_residual,
)
from agencitylab.fields.numerics import UniformRectilinearGrid
from agencitylab.scientific_status import ScientificStatus


def test_coherent_package_is_research_status():
    assert SCIENTIFIC_STATUS is ScientificStatus.RESEARCH


def test_domain_wall_profile_matches_rescaled_real_sector_kink():
    lambda_ = 2.0
    mu = 0.5
    v = np.sqrt(lambda_ / mu)
    x = np.linspace(-12.0, 12.0, 481)

    phi = domain_wall_profile(x, lambda_=lambda_, mu=mu, center=0.0)

    assert np.isrealobj(phi)
    assert phi[x.size // 2] == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(phi, -phi[::-1], atol=2e-14, rtol=0.0)
    assert phi[0] == pytest.approx(-v, abs=2e-10)
    assert phi[-1] == pytest.approx(v, abs=2e-10)


def test_domain_wall_center_and_orientation_are_explicit():
    x = np.linspace(-5.5, 6.5, 241)
    center = 0.5
    kink = domain_wall_profile(x, lambda_=1.5, mu=0.75, center=center, orientation=1)
    antikink = domain_wall_profile(x, lambda_=1.5, mu=0.75, center=center, orientation=-1)
    center_index = int(np.argmin(np.abs(x - center)))

    assert x[center_index] == pytest.approx(center)
    assert kink[center_index] == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(antikink, -kink, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("lambda_", "mu"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_domain_wall_requires_broken_real_vacua(lambda_, mu):
    with pytest.raises(ValueError):
        domain_wall_profile(np.linspace(-1.0, 1.0, 9), lambda_=lambda_, mu=mu)


def test_domain_wall_rejects_non_real_sector_and_invalid_orientation():
    with pytest.raises(ValueError, match="real"):
        domain_wall_profile(
            np.array([-1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]),
            lambda_=1.0,
            mu=1.0,
        )
    with pytest.raises(ValueError, match="orientation"):
        domain_wall_profile(np.linspace(-1.0, 1.0, 9), lambda_=1.0, mu=1.0, orientation=0)


def _wall_residual_error(n_points: int) -> float:
    x = np.linspace(-6.0, 6.0, n_points)
    grid = UniformRectilinearGrid(axes=(x,))
    phi = domain_wall_profile(x, lambda_=2.0, mu=0.5)
    residual = domain_wall_residual(phi, grid, lambda_=2.0, mu=0.5)
    margin = n_points // 4
    return float(np.max(np.abs(residual[margin:-margin])))


def test_domain_wall_stationary_residual_converges_at_second_order():
    coarse = _wall_residual_error(201)
    fine = _wall_residual_error(401)

    assert fine < coarse
    assert coarse / fine > 3.5


def test_domain_wall_residual_validates_one_dimensional_real_field():
    grid_2d = UniformRectilinearGrid(shape=(8, 8), spacings=(0.1, 0.1))
    with pytest.raises(ValueError, match="one-dimensional"):
        domain_wall_residual(np.zeros((8, 8)), grid_2d, lambda_=1.0, mu=1.0)

    grid_1d = UniformRectilinearGrid(shape=(8,), spacings=(0.1,))
    with pytest.raises(ValueError, match="real"):
        domain_wall_residual(np.zeros(8, dtype=complex), grid_1d, lambda_=1.0, mu=1.0)


def test_vortex_field_on_cartesian_grid_has_expected_shape_and_core():
    axis = np.linspace(-4.0, 4.0, 81)
    phi = vortex_field(
        x=axis,
        y=axis,
        winding=1,
        lambda_=2.0,
        mu=0.5,
        radial_profile=lambda r: np.tanh(r),
    )

    assert phi.shape == (81, 81)
    assert np.iscomplexobj(phi)
    assert phi[40, 40] == 0.0j


def test_zero_winding_vortex_reference_reduces_to_radial_amplitude():
    x = np.linspace(-2.0, 2.0, 17)
    y = np.linspace(-3.0, 3.0, 19)
    v = np.sqrt(2.0 / 0.5)
    phi = vortex_field(
        x=x,
        y=y,
        winding=0,
        lambda_=2.0,
        mu=0.5,
        radial_profile=lambda r: np.ones_like(r),
    )

    np.testing.assert_allclose(phi, v + 0.0j, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("winding", [0, 1, -1, 2])
def test_phase_winding_recovers_integer_vortex_charge(winding):
    theta = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    radius = np.full_like(theta, 3.0)
    phi = vortex_field(
        r=radius,
        theta=theta,
        winding=winding,
        lambda_=1.5,
        mu=0.75,
        radial_profile=np.ones_like(radius),
    )

    assert phase_winding(phi) == pytest.approx(float(winding), abs=1e-12)


def test_vortex_winding_and_modulus_are_invariant_under_global_u1_phase():
    theta = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    radius = np.full_like(theta, 2.5)
    phi = vortex_field(
        r=radius,
        theta=theta,
        winding=2,
        lambda_=2.0,
        mu=1.0,
        radial_profile=np.ones_like(radius),
    )
    rotated = phi * np.exp(1j * 0.731)

    assert phase_winding(rotated) == pytest.approx(phase_winding(phi), abs=1e-12)
    np.testing.assert_allclose(np.abs(rotated), np.abs(phi), rtol=1e-14, atol=1e-14)


def test_vortex_winding_sign_tracks_orientation():
    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    radius = np.full_like(theta, 4.0)
    positive = vortex_field(
        r=radius,
        theta=theta,
        winding=1,
        lambda_=1.0,
        mu=1.0,
        radial_profile=np.ones_like(radius),
    )
    negative = vortex_field(
        r=radius,
        theta=theta,
        winding=-1,
        lambda_=1.0,
        mu=1.0,
        radial_profile=np.ones_like(radius),
    )

    assert phase_winding(positive) == pytest.approx(1.0, abs=1e-12)
    assert phase_winding(negative) == pytest.approx(-1.0, abs=1e-12)


def test_vortex_requires_integer_winding_and_user_supplied_core_boundary():
    with pytest.raises(TypeError, match="integer"):
        vortex_field(
            r=np.array([1.0, 2.0, 3.0]),
            theta=np.array([0.0, 1.0, 2.0]),
            winding=1.5,
            lambda_=1.0,
            mu=1.0,
            radial_profile=np.ones(3),
        )

    with pytest.raises(ValueError, match=r"f\(0\)"):
        vortex_field(
            r=np.array([0.0, 1.0, 2.0]),
            theta=np.array([0.0, 0.0, 0.0]),
            winding=1,
            lambda_=1.0,
            mu=1.0,
            radial_profile=np.ones(3),
        )


def test_vortex_radial_residual_matches_exact_n0_uniform_vacuum():
    r = np.linspace(0.0, 5.0, 101)
    f = np.ones_like(r)
    residual = vortex_radial_residual(r, f, winding=0, lambda_=2.0)

    assert residual.shape == (99,)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12, rtol=0.0)


def test_vortex_radial_residual_is_finite_for_user_reference_profile_away_from_core():
    r = np.linspace(0.0, 8.0, 321)
    f = np.tanh(r / 1.7)
    residual = vortex_radial_residual(r, f, winding=1, lambda_=1.0)

    assert residual.shape == (319,)
    assert np.all(np.isfinite(residual))


def test_vortex_radial_residual_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        vortex_radial_residual(
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.array([0.0, 0.5, 0.8, 1.0]),
            winding=1,
            lambda_=1.0,
        )
    with pytest.raises(ValueError, match="lambda"):
        vortex_radial_residual(
            np.linspace(0.0, 1.0, 8),
            np.linspace(0.0, 1.0, 8),
            winding=1,
            lambda_=0.0,
        )


def test_phase_winding_rejects_contour_crossing_exact_zero():
    with pytest.raises(ValueError, match="undefined"):
        phase_winding(np.array([1.0 + 0.0j, 0.0j, -1.0 + 0.0j, 1.0j]))


def test_field_zero_mask_uses_only_exact_zero_without_user_tolerance():
    field = np.array([0.0j, 1e-12 + 0.0j, 1.0 + 1.0j])
    exact = field_zero_mask(field)
    near = field_zero_mask(field, tolerance=1e-10)

    np.testing.assert_array_equal(exact, np.array([True, False, False]))
    np.testing.assert_array_equal(near, np.array([True, True, False]))


def test_field_zero_mask_rejects_invalid_user_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        field_zero_mask(np.array([0.0j, 1.0j]), tolerance=-1.0)
