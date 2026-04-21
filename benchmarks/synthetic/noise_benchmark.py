import numpy as np
from agencitylab.api.compute import compute_agencity
import time

def run():
    xi = np.linspace(0, 100, 5000)
    u = np.random.randn(len(xi))

    t0 = time.time()
    compute_agencity(xi, u)
    dt = time.time() - t0

    print("Noise benchmark time:", dt)
    return dt

if __name__ == "__main__":
    run()
