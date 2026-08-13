import numpy as np
import pytest

from agencitylab.core.crm import causal_moving_correlation


def _direct_pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    a = a / np.max(np.abs(a))
    b = b / np.max(np.abs(b))
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))


def _reference_crm(signal, width, other=None):
    x = np.asarray(signal, dtype=float)
    y = x if other is None else np.asarray(other, dtype=float)
    out = np.zeros(x.size, dtype=float)
    for end in range(2 * width - 1, x.size):
        recent = x[end - width + 1 : end + 1]
        previous = y[end - 2 * width + 1 : end - width + 1]
        out[end] = _direct_pearson(recent, previous)
    return out


@pytest.mark.parametrize("width", [2, 3, 8, 31])
def test_vectorized_auto_crm_matches_direct_reference(width):
    rng = np.random.default_rng(8341 + width)
    signal = rng.normal(size=2048)
    axis = np.arange(signal.size, dtype=float) * 0.125

    actual = causal_moving_correlation(
        signal,
        tau=1.0,
        axis=axis,
        window=width * 0.125,
    )
    expected = _reference_crm(signal, width)

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("width", [2, 7, 23])
def test_vectorized_cross_crm_matches_direct_reference(width):
    rng = np.random.default_rng(1907 + width)
    signal = rng.normal(size=1536)
    other = 0.3 * signal + rng.normal(scale=0.5, size=signal.size)
    axis = np.arange(signal.size, dtype=float)

    actual = causal_moving_correlation(
        signal,
        tau=1.0,
        axis=axis,
        other=other,
        window=float(width),
    )
    expected = _reference_crm(signal, width, other=other)

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_piecewise_constant_windows_keep_exact_zero_variance_convention():
    signal = np.array(
        [1.0] * 4
        + [2.0] * 4
        + [1.0, 2.0, 3.0, 4.0]
        + [1.0, 2.0, 3.0, 4.0]
    )
    axis = np.arange(signal.size, dtype=float)

    actual = causal_moving_correlation(signal, 4.0, axis=axis)
    expected = _reference_crm(signal, 4)

    np.testing.assert_array_equal(actual[:11], expected[:11])
    assert actual[-1] == pytest.approx(1.0)


def test_numerical_fallback_preserves_tiny_local_structure_near_huge_values():
    large = np.array([1.0, 2.0, 3.0, 4.0]) * 1e300
    tiny = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-300
    signal = np.concatenate([large, tiny, tiny])
    axis = np.arange(signal.size, dtype=float)

    actual = causal_moving_correlation(signal, 4.0, axis=axis)

    assert actual[-1] == pytest.approx(1.0)


def test_long_signal_crm_is_finite_bounded_and_shape_preserving():
    size = 100_000
    axis = np.arange(size, dtype=float)
    signal = np.sin(0.017 * axis) + 0.2 * np.cos(0.031 * axis)

    result = causal_moving_correlation(signal, 128.0, axis=axis)

    assert result.shape == signal.shape
    assert np.all(np.isfinite(result))
    assert np.all(np.abs(result) <= 1.0)
    np.testing.assert_array_equal(result[:255], 0.0)
