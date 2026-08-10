import numpy as np

from agencitylab import compute_agencity


def test_full_pipeline():
    xi = np.linspace(0, 10, 200)
    u = np.sin(xi)

    result = compute_agencity(u=u, xi=xi)

    assert result.b.shape == xi.shape
    assert result.beta.shape == xi.shape
