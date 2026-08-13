import numpy as np
import pytest

from agencitylab import compute_agencity
from agencitylab.backends import (
    backend_capabilities,
    causal_moving_correlation_numba,
    causal_moving_correlation_numpy,
    select_backend,
)


def test_backend_capabilities_distinguish_reference_and_experimental_scope():
    capabilities = backend_capabilities()

    assert capabilities["numpy"]["status"] == "stable"
    assert capabilities["numpy"]["canonical_pipeline"] is True
    assert capabilities["numpy"]["available"] is True

    assert capabilities["numba"]["status"] == "experimental"
    assert capabilities["numba"]["canonical_pipeline"] is False
    assert capabilities["jax"]["status"] == "experimental"
    assert capabilities["jax"]["canonical_pipeline"] is False


def test_selected_numpy_backend_carries_capability_metadata():
    backend = select_backend("numpy")

    assert backend["name"] == "numpy"
    assert backend["status"] == "stable"
    assert backend["canonical_pipeline"] is True
    assert callable(backend["causal_moving_correlation"])


def test_optional_crm_wrappers_do_not_reclassify_tiny_variance_as_zero():
    values = 1e-20 * np.array([0.0, 1.0, 0.0, 1.0])

    numpy_result = causal_moving_correlation_numpy(values, window=2)
    numba_result = causal_moving_correlation_numba(values, window=2)

    np.testing.assert_allclose(
        numpy_result[-1],
        1.0,
        rtol=0.0,
        atol=4.0 * np.finfo(float).eps,
    )
    np.testing.assert_array_equal(numba_result, numpy_result)


def test_compute_api_is_backend_agnostic_and_rejects_pre_1_0_config_keyword():
    xi = np.linspace(0.0, 4.0, 33)
    result = compute_agencity(
        u=np.sin(xi),
        xi=xi,
        A_ref=1.0,
        tau=0.5,
        w=0.5,
        P_c=2.0,
    )

    assert not hasattr(result, "config")
    with pytest.raises(TypeError):
        compute_agencity(
            u=np.sin(xi),
            xi=xi,
            A_ref=1.0,
            tau=0.5,
            w=0.5,
            P_c=2.0,
            config={"backend": "numpy"},
        )
