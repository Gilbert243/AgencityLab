import numpy as np
import pytest

from agencitylab import PhysicalParameterError, compute_agencity
from agencitylab.core.agencity import agencity, agencity_criteria, compute_full_agencity
from agencitylab.core.coherence import coherence_diagnostic
from agencitylab.core.contrast import compute_contrast
from agencitylab.core.normalization import normalize_signal


def _sample_signal():
    xi = np.arange(0.0, 8.1, 0.1)
    return xi, np.sin(0.7 * xi) + 0.1 * np.sin(1.9 * xi)


def test_w_and_tau_are_independent_and_explicit_w_is_preserved():
    xi, u = _sample_signal()
    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=2.0,
        tau=1.0,
        w=0.5,
        P_c=1.0,
    )
    assert result.tau == 1.0
    assert result.memory_window == 0.5
    assert result.metadata.extra["memory_window_mode"] == "explicit"
    assert result.metadata.extra["memory_window_convention"] == (
        "w was supplied explicitly and preserved"
    )


def test_omitted_w_records_explicit_fallback_convention():
    xi, u = _sample_signal()
    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=1.0,
        tau=1.0,
        P_c=1.0,
    )
    assert result.memory_window == result.tau == 1.0
    assert result.metadata.extra["memory_window_convention"] == (
        "w was unspecified; implementation convention w = tau was used"
    )


def test_public_canonical_pipeline_never_invents_A_ref_from_signal_statistics():
    xi, u = _sample_signal()
    with pytest.raises(PhysicalParameterError, match="A_ref"):
        compute_agencity(u=u, xi=xi, tau=1.0, w=0.5, P_c=1.0)

    u_star, ref = normalize_signal(u, A_ref=4.0, method="canonical")
    np.testing.assert_array_equal(u_star, u / 4.0)
    assert ref == 4.0


def test_contrast_uses_eulers_number_not_one_or_epsilon():
    D = np.array([0.25, 2.0, 7.0])
    S = np.array([1.5, 0.5, 3.0])
    actual = compute_contrast(D, S)
    expected = np.log((np.exp(1.0) + D) / (np.exp(1.0) + S))
    historical_one = np.log((1.0 + D) / (1.0 + S))
    np.testing.assert_array_equal(actual, expected)
    assert not np.allclose(actual, historical_one)


def test_zero_characteristic_power_is_exactly_valid_for_scalar_operator():
    beta = np.array([1.0 + 2.0j, -0.5j, 3.0 - 1.0j])
    np.testing.assert_array_equal(agencity(beta, 0.0), 0.0j)


def test_zero_and_partly_zero_sampled_power_preserve_beta_and_zero_flux_locally():
    xi, u = _sample_signal()
    power = np.ones_like(xi)
    power[20:40] = 0.0
    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=1.0,
        tau=1.0,
        w=0.5,
        P_c=power,
    )
    np.testing.assert_array_equal(result.b[20:40], 0.0j)
    np.testing.assert_allclose(result.b, power * result.beta)
    assert np.any(np.abs(result.beta[20:40]) > 0.0)


@pytest.mark.parametrize("bad_power", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_scalar_power_is_rejected_without_epsilon_repair(bad_power):
    xi, u = _sample_signal()
    with pytest.raises(PhysicalParameterError):
        compute_agencity(
            u=u,
            xi=xi,
            A_ref=1.0,
            tau=1.0,
            w=0.5,
            P_c=bad_power,
        )


def test_zero_structure_remains_exact_null_branch_far_below_common_epsilons():
    xi = np.arange(0.0, 8.1, 0.1)
    result = compute_agencity(
        u=np.ones_like(xi),
        xi=xi,
        A_ref=1.0,
        tau=1.0,
        w=0.5,
        P_c=0.0,
    )
    np.testing.assert_array_equal(result.S, 0.0)
    np.testing.assert_array_equal(result.U, 0.0j)
    np.testing.assert_array_equal(result.beta, 0.0j)
    np.testing.assert_array_equal(result.b, 0.0j)


def test_legacy_full_pipeline_delegates_to_reference_and_accepts_w_not_equal_tau():
    xi, u = _sample_signal()
    reference = compute_agencity(
        u=u,
        xi=xi,
        A_ref=1.0,
        tau=1.0,
        w=0.5,
        P_c=2.0,
    )
    with pytest.warns(DeprecationWarning, match="legacy compatibility wrapper"):
        legacy = compute_full_agencity(
            xi,
            u,
            A_ref=1.0,
            tau=1.0,
            w=0.5,
            P_c=2.0,
        )
    assert legacy["canonical_reference"] == "agencitylab.compute_agencity"
    assert "not an independent canonical pipeline" in legacy["status"]
    assert legacy["w"] == 0.5
    for name in ("M", "O", "D", "S", "J", "U", "beta", "b"):
        np.testing.assert_allclose(legacy[name], getattr(reference, name))


def test_real_agencity_is_absent_from_reference_canonical_result():
    xi, u = _sample_signal()
    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=1.0,
        tau=1.0,
        w=0.5,
        P_c=1.0,
    )
    assert result.analysis == {}
    assert not hasattr(result, "real_agencity")

    with pytest.warns(DeprecationWarning, match="legacy diagnostic"):
        legacy = agencity_criteria(result.M, result.O, result.S, result.b)
    assert legacy["status"].startswith("legacy diagnostic")

    with pytest.warns(DeprecationWarning, match="legacy diagnostic"):
        coherence_diagnostic(result.M, result.O)
