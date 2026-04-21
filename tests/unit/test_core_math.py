import numpy as np

def test_basic_derivative():
    x = np.linspace(0, 10, 100)
    y = x**2
    dy = np.gradient(y, x)
    assert np.allclose(dy[50], 2*x[50], atol=1e-1)

def test_tanh_bounds():
    x = np.linspace(-10, 10, 1000)
    y = np.tanh(x)
    assert np.all(y <= 1.0)
    assert np.all(y >= -1.0)
