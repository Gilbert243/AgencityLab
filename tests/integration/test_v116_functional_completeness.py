"""Cross-layer gates for AgencityLab 1.1.6 functional completeness.

These tests verify source transcription and separation from the canonical scalar
engine. They are not a falsification protocol for the Theory of Agencity.
"""

from __future__ import annotations

import numpy as np

import agencitylab
from agencitylab.analysis import (
    multiplicative_power_perturbation,
    recoverable_agencity_signature,
)
from agencitylab.extensions import (
    INTENSITY_ALTERNATIVES_STATUS,
    printed_offset_ratio_candidate,
    sum_intensity,
)
from agencitylab.scientific_status import ScientificStatus


def test_historical_intensity_candidates_do_not_replace_canonical_contrast() -> None:
    xi = np.arange(128.0)
    result = agencitylab.compute_agencity(
        u=np.sin(0.15 * xi) + 0.1 * np.cos(0.31 * xi),
        xi=xi,
        A_ref=1.0,
        tau=6.0,
        w=5.0,
        P_c=2.0,
    )

    expected_j = np.log((np.e + result.D) / (np.e + result.S))
    np.testing.assert_allclose(result.J, expected_j)
    np.testing.assert_allclose(result.b, result.P_c * result.beta)

    historical = sum_intensity(result.X_star, result.A_star, result.M, result.O)
    offset = printed_offset_ratio_candidate(
        result.X_star,
        result.A_star,
        result.M,
        result.O,
    )
    assert historical.shape == result.J.shape
    assert offset.shape == result.J.shape
    assert INTENSITY_ALTERNATIVES_STATUS is ScientificStatus.EXPERIMENTAL


def test_recoverable_inverse_signature_roundtrips_only_macroscopic_state() -> None:
    xi = np.arange(128.0)
    result = agencitylab.compute_agencity(
        u=np.sin(0.11 * xi),
        xi=xi,
        A_ref=1.0,
        tau=7.0,
        w=6.0,
        P_c=3.0,
    )

    signature = recoverable_agencity_signature(result.b, result.P_c)
    np.testing.assert_allclose(signature["beta"], result.beta)
    np.testing.assert_allclose(signature["absolute_contrast"], np.abs(result.beta))

    defined = signature["direction_defined"]
    np.testing.assert_allclose(
        signature["orientation_mod_pi"][defined],
        np.mod(result.theta[defined], np.pi),
    )


def test_power_perturbation_remains_external_to_intrinsic_beta() -> None:
    beta = np.array([0.2 + 0.3j, -0.4 + 0.1j])
    p_c0 = 5.0
    b0 = p_c0 * beta
    epsilon = np.array([0.1, -0.2])

    perturbed, delta = multiplicative_power_perturbation(b0, epsilon)
    np.testing.assert_allclose(delta, epsilon * b0)
    np.testing.assert_allclose(perturbed, p_c0 * (1.0 + epsilon) * beta)
