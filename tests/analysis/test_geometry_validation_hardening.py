import numpy as np
import pytest

from agencitylab.analysis.geometry import (
    curvature,
    geometric_summary,
    net_phase_turns,
    winding_number,
)


def test_straight_line_curvature_is_scale_and_rotation_invariant():
    xi = np.linspace(-1.0, 1.0, 1001)
    base = (2.0 + 3.0j) + (1.2 - 0.7j) * xi

    for scale in (1e-12, 1.0, 1e12):
        for phase in (0.0, 0.73):
            beta = scale * np.exp(1j * phase) * base
            kappa = curvature(beta, xi)
            finite = np.isfinite(kappa)
            assert np.any(finite)
            np.testing.assert_array_equal(kappa[finite], 0.0)


def test_stationary_trajectory_and_exact_turning_point_are_undefined():
    xi = np.linspace(-1.0, 1.0, 1001)
    constant = np.full(xi.size, 2.0 - 4.0j)
    assert np.all(np.isnan(curvature(constant, xi)))

    cubic_line = (1.0 + 2.0j) * xi**3
    centre = xi.size // 2
    kappa = curvature(cubic_line, xi)
    assert np.isnan(kappa[centre])
    finite = np.isfinite(kappa)
    assert np.any(finite)
    np.testing.assert_array_equal(kappa[finite], 0.0)


def test_circle_reference_curvature_remains_correct():
    xi = np.linspace(0.0, 2.0 * np.pi, 1001)
    radius = 2.5
    beta = radius * np.exp(1j * xi)

    kappa = curvature(beta, xi)

    np.testing.assert_allclose(
        np.median(kappa[5:-5]),
        1.0 / radius,
        rtol=2e-3,
        atol=2e-3,
    )


def test_sinusoidal_parameterisation_of_a_line_does_not_explode():
    xi = np.linspace(0.0, 8.0 * np.pi, 4001)
    beta = (0.3 - 0.2j) + (1.1 + 0.4j) * np.sin(xi)

    kappa = curvature(beta, xi)

    finite = np.isfinite(kappa)
    assert np.any(finite)
    assert np.any(~finite)
    np.testing.assert_array_equal(kappa[finite], 0.0)


def test_contiguous_valid_mask_matches_explicit_phase_slice():
    theta = np.linspace(0.0, 6.0 * np.pi, 301)
    mask = np.zeros(theta.size, dtype=bool)
    mask[40:280] = True

    direct = net_phase_turns(theta, valid_mask=mask)
    sliced = net_phase_turns(theta[40:280])

    assert direct == pytest.approx(sliced)


def test_geometric_summary_accepts_leading_and_trailing_invalid_phase_samples():
    theta = np.linspace(0.0, 6.0 * np.pi, 301)
    beta = np.exp(1j * theta)
    mask = np.zeros(theta.size, dtype=bool)
    mask[40:280] = True

    summary = geometric_summary(
        beta,
        xi=np.arange(beta.size, dtype=float),
        theta=theta,
        valid_mask=mask,
        winding_closed=False,
    )

    assert summary["net_phase_turns"] == pytest.approx(net_phase_turns(theta[40:280]))
    assert np.isnan(summary["winding"]["winding_number"])


def test_discontinuous_valid_mask_remains_undefined():
    theta = np.linspace(0.0, 2.0 * np.pi, 101)
    mask = np.ones(theta.size, dtype=bool)
    mask[50] = False

    assert np.isnan(net_phase_turns(theta, valid_mask=mask))
    assert np.isnan(winding_number(theta, valid_mask=mask))
