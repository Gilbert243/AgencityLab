import numpy as np

from agencitylab import compute_agencity


def test_full_pipeline():
    xi = np.linspace(0.0, 10.0, 101)
    u = np.sin(xi)
    result = compute_agencity(u=u, xi=xi, A_ref=1.0, tau=2.0, P_c=1.0)
    assert result.b.shape == xi.shape
    assert result.beta.shape == xi.shape
    np.testing.assert_allclose(result.b, result.P_c * result.beta)
