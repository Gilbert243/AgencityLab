from __future__ import annotations

import numpy as np
import pytest

from agencitylab.analysis import (
    logarithmic_contrast_offset_sensitivity,
    multiplicative_power_perturbation,
    recoverable_agencity_signature,
)


def test_logarithmic_contrast_offset_sensitivity_matches_chapter_4() -> None:
    D = np.array([0.0, 1.0, 3.0])
    S = np.array([1.0, 1.0, 0.5])
    expected = (S - D) / ((np.e + D) * (np.e + S))
    np.testing.assert_allclose(logarithmic_contrast_offset_sensitivity(D, S), expected)
    assert logarithmic_contrast_offset_sensitivity(2.0, 2.0) == pytest.approx(0.0)


def test_logarithmic_contrast_offset_sensitivity_rejects_invalid_intensities() -> None:
    with pytest.raises(ValueError):
        logarithmic_contrast_offset_sensitivity(-1.0, 0.0)
    with pytest.raises(ValueError):
        logarithmic_contrast_offset_sensitivity(1.0, np.nan)


def test_multiplicative_power_perturbation_matches_chapter_10_exactly() -> None:
    b0 = np.array([1.0 + 2.0j, -3.0j])
    epsilon = np.array([0.1, -0.25])
    perturbed, delta = multiplicative_power_perturbation(b0, epsilon)
    np.testing.assert_allclose(delta, epsilon * b0)
    np.testing.assert_allclose(perturbed, b0 + epsilon * b0)


def test_recoverable_signature_returns_absolute_contrast_and_direction_modulo_sign() -> None:
    theta = np.array([0.2, 1.1])
    J = np.array([0.7, -1.4])
    beta = J * np.exp(1j * theta)
    P_c = np.array([2.0, 4.0])
    b = P_c * beta

    signature = recoverable_agencity_signature(b, P_c)
    np.testing.assert_allclose(signature["beta"], beta)
    np.testing.assert_allclose(signature["absolute_contrast"], np.abs(J))
    np.testing.assert_allclose(signature["orientation_mod_pi"], np.mod(theta, np.pi))
    assert np.all(signature["direction_defined"])


def test_inverse_direction_is_undefined_at_zero_flux() -> None:
    signature = recoverable_agencity_signature(0.0j, 2.0)
    assert signature["beta"] == 0.0j
    assert signature["absolute_contrast"] == 0.0
    assert np.isnan(signature["orientation_mod_pi"])
    assert signature["direction_defined"] is False


def test_inverse_refuses_zero_characteristic_power() -> None:
    with pytest.raises(ValueError, match="P_c must be strictly positive"):
        recoverable_agencity_signature(0.0j, 0.0)
