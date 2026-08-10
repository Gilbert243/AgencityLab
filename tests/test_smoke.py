import numpy as np
from agencitylab.api import compute_agencity

def test_compute_smoke():
    t = np.linspace(0, 10, 100)
    u = np.sin(t)
    r = compute_agencity(t, u)
    assert r.b.shape == t.shape
