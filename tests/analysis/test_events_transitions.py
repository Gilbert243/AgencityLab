import numpy as np

from agencitylab.analysis.events import detect_dynamic_peaks
from agencitylab.analysis.transitions import (
    critical_surface_crossings,
    detect_agencity_zeros,
    detect_theta_jumps,
)


def test_dynamic_peaks_operate_on_D():
    D = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0])

    np.testing.assert_array_equal(detect_dynamic_peaks(D), np.array([1, 3, 5]))
    np.testing.assert_array_equal(
        detect_dynamic_peaks(D, prominence=1.5),
        np.array([3]),
    )


def test_unfiltered_dynamic_peaks_keep_flat_peak_midpoint_convention():
    D = np.array([0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    np.testing.assert_array_equal(detect_dynamic_peaks(D), np.array([1, 5]))


def test_agencity_zeros_follow_S_or_J_condition_exactly():
    S = np.array([0.0, 1.0, 1.0, 2.0])
    J = np.array([3.0, 0.0, 1e-14, -1.0])

    np.testing.assert_array_equal(detect_agencity_zeros(S, J), np.array([0, 1]))
    np.testing.assert_array_equal(
        detect_agencity_zeros(S, J, atol=1e-12),
        np.array([0, 1, 2]),
    )


def test_critical_surface_crossings_use_exact_or_sign_change():
    D = np.array([0.0, 2.0, 1.0, 3.0])
    S = np.array([1.0, 1.0, 1.0, 2.0])

    np.testing.assert_array_equal(
        critical_surface_crossings(D, S),
        np.array([0, 2]),
    )


def test_theta_jump_detector_respects_branch_cut():
    theta = np.array([3.10, -3.10, -2.90, 0.0])

    indices = detect_theta_jumps(theta, threshold=1.0)

    np.testing.assert_array_equal(indices, np.array([2]))
