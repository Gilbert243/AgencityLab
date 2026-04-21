import numpy as np
import pytest

@pytest.fixture
def simple_signal():
    xi = np.linspace(0, 10, 100)
    u = np.sin(xi)
    return xi, u
