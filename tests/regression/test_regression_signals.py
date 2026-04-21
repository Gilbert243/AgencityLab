import numpy as np
from agencitylab.api.compute import compute_agencity

def test_sine_signal_regression():
    xi = np.linspace(0, 2*np.pi, 500)
    u = np.sin(xi)

    result = compute_agencity(xi, u)

    assert abs(np.mean(result.b)) < 10
