import numpy as np
import pytest

from agencitylab.analysis.signature import agencity_signature


def test_multiscale_signature_does_not_invent_regime_threshold():
    result = agencity_signature(
        tau=np.array([1.0, 2.0, 4.0]),
        beta_mean=np.array([1.0, 2.0, 4.0]),
    )

    np.testing.assert_allclose(result["slope"], 1.0, atol=1e-12)
    assert result["regime"] == "undetermined"
    assert result["slope_threshold"] is None


def test_multiscale_signature_interpretation_requires_explicit_threshold():
    result = agencity_signature(
        tau=np.array([1.0, 2.0, 4.0]),
        beta_mean=np.array([1.0, 2.0, 4.0]),
        slope_threshold=0.2,
    )

    assert result["regime"] == "amplifying"
    assert result["interpretation_status"] == "diagnostic threshold configured"


def test_multiscale_signature_rejects_nonpositive_data_instead_of_epsilon_substitution():
    with pytest.raises(ValueError, match="at least two"):
        agencity_signature(
            tau=np.array([1.0, 2.0]),
            beta_mean=np.array([0.0, 1.0]),
        )
