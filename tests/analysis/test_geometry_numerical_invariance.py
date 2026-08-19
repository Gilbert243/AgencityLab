import numpy as np

from agencitylab.analysis.geometry import curvature


def test_straight_line_curvature_survives_large_resolvable_translations():
    xi = np.linspace(-1.0, 1.0, 1001)
    direction = 2.0 - 0.75j

    for magnitude in (0.0, 1.0, 1e3, 1e6, 1e9):
        offset = magnitude * (1.0 + 0.3j)
        beta = offset + direction * xi
        kappa = curvature(beta, xi)

        finite = np.isfinite(kappa)
        assert np.any(finite)
        assert not np.any(np.isinf(kappa))
        np.testing.assert_array_equal(kappa[finite], 0.0)


def test_coordinate_resolution_floor_does_not_flatten_resolved_curvature():
    xi = np.linspace(-1.0, 1.0, 1001)
    direction = 2.0 - 0.75j
    normal = 1j * direction / abs(direction)
    bend = 1e-4

    reference = direction * xi + bend * normal * xi**2
    translated = 1e6 * (1.0 + 0.3j) + reference

    reference_kappa = curvature(reference, xi)
    translated_kappa = curvature(translated, xi)
    valid = np.isfinite(reference_kappa) & np.isfinite(translated_kappa)

    assert np.any(valid)
    assert np.any(np.abs(reference_kappa[valid]) > 0.0)
    assert np.any(np.abs(translated_kappa[valid]) > 0.0)
    np.testing.assert_allclose(
        np.median(translated_kappa[valid]),
        np.median(reference_kappa[valid]),
        rtol=5e-2,
        atol=1e-8,
    )


def test_circle_curvature_is_scale_safe_near_float_extremes():
    xi = np.linspace(0.0, 2.0 * np.pi, 2001)

    for radius in (1e-300, 1e-200, 1e-150, 1e150, 1e200, 1e300):
        beta = radius * np.exp(1j * xi)
        kappa = curvature(beta, xi)

        finite = np.isfinite(kappa)
        assert np.all(finite)
        assert not np.any(np.isinf(kappa))
        np.testing.assert_allclose(
            np.median(kappa[8:-8]),
            1.0 / radius,
            rtol=5e-9,
            atol=0.0,
        )


def test_circle_curvature_is_stable_under_extreme_axis_rescaling():
    phase = np.linspace(0.0, 2.0 * np.pi, 2001)
    beta = np.exp(1j * phase)

    for axis_scale in (1e-150, 1e-100, 1.0, 1e100, 1e150):
        kappa = curvature(beta, axis_scale * phase)

        finite = np.isfinite(kappa)
        assert np.all(finite)
        assert not np.any(np.isinf(kappa))
        np.testing.assert_allclose(
            np.median(kappa[8:-8]),
            1.0,
            rtol=5e-9,
            atol=5e-12,
        )


def test_translated_circle_remains_curved_when_geometry_is_resolved():
    xi = np.linspace(0.0, 2.0 * np.pi, 2001)
    beta = 1e8 * (1.0 + 0.3j) + np.exp(1j * xi)

    kappa = curvature(beta, xi)
    finite = np.isfinite(kappa)

    assert np.all(finite)
    assert not np.any(np.isinf(kappa))
    assert np.any(kappa != 0.0)
    np.testing.assert_allclose(
        np.median(kappa[8:-8]),
        1.0,
        rtol=5e-3,
        atol=5e-6,
    )
