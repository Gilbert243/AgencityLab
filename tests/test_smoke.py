import numpy as np

from agencitylab import compute_agencity


def test_compute_smoke():
    xi = np.linspace(0, 10, 100)
    u = np.sin(xi)
    result = compute_agencity(u=u, xi=xi)
    assert result.b.shape == xi.shape
