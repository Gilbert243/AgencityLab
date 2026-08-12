from __future__ import annotations

import numpy as np
import pytest

from agencitylab.extensions import (
    INTENSITY_ALTERNATIVES_STATUS,
    printed_offset_ratio_candidate,
    raw_ratio_intensity,
    sum_intensity,
    sum_log_intensity,
)
from agencitylab.scientific_status import ScientificStatus


def test_historical_intensity_alternatives_are_experimental_only() -> None:
    assert INTENSITY_ALTERNATIVES_STATUS is ScientificStatus.EXPERIMENTAL


def test_sum_form_matches_equation_14_1() -> None:
    X = np.array([1.0, -2.0])
    A = np.array([3.0, 0.5])
    M = np.array([-0.25, 0.75])
    O = np.array([0.5, -0.1])
    expected = np.abs(X) + np.abs(A * X) + np.abs(M) + np.abs(O)
    np.testing.assert_allclose(sum_intensity(X, A, M, O), expected)
    np.testing.assert_allclose(sum_log_intensity(X, A, M, O), np.log(expected))


def test_sum_log_form_exposes_exact_rest_singularity_without_epsilon() -> None:
    assert sum_intensity(0.0, 0.0, 0.0, 0.0) == 0.0
    assert sum_log_intensity(0.0, 0.0, 0.0, 0.0) == -np.inf


def test_raw_ratio_matches_equation_14_2_and_preserves_singularities() -> None:
    assert raw_ratio_intensity(2.0, 3.0, 0.5, -0.5) == pytest.approx(8.0)
    assert raw_ratio_intensity(1.0, 0.0, 0.0, 0.0) == np.inf
    assert np.isnan(raw_ratio_intensity(0.0, 0.0, 0.0, 0.0))


def test_printed_offset_ratio_candidate_uses_fixed_natural_e() -> None:
    X, A, M, O = 2.0, 3.0, 0.5, -0.5
    numerator = abs(X) + abs(A * X)
    denominator = np.e + abs(M) + abs(O)
    expected = np.e + numerator / denominator
    assert printed_offset_ratio_candidate(X, A, M, O) == pytest.approx(expected)


def test_intensity_alternatives_reject_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        sum_intensity(np.nan, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        raw_ratio_intensity(1.0, np.inf, 1.0, 1.0)
