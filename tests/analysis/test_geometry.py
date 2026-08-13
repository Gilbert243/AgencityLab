import numpy as np

from agencitylab.analysis.geometry import curvature, winding_number


def test_curvature_is_zero_for_rectilinear_beta_trajectory():
    xi = np.linspace(0.0, 10.0, 401)
    beta = xi.astype(complex)

    kappa = curvature(beta, xi)

    finite = np.isfinite(kappa)
    assert np.any(finite)
    np.testing.assert_allclose(kappa[finite], 0.0, atol=1e-10)


def test_curvature_of_unit_circle_converges_to_one():
    xi = np.linspace(0.0, 2.0 * np.pi, 1001)
    beta = np.exp(1j * xi)

    kappa = curvature(beta, xi)

    np.testing.assert_allclose(np.median(kappa[5:-5]), 1.0, rtol=2e-3, atol=2e-3)


def test_winding_number_uses_structural_orientation():
    theta = np.linspace(0.0, 2.0 * np.pi, 1001)

    np.testing.assert_allclose(winding_number(theta), 1.0, atol=1e-12)


def test_winding_is_undefined_across_structural_zero():
    theta = np.linspace(0.0, 2.0 * np.pi, 101)
    valid = np.ones(theta.size, dtype=bool)
    valid[50] = False

    assert np.isnan(winding_number(theta, valid_mask=valid))
