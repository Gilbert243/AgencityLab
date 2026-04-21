import numpy as np
from agencitylab.api.compute import compute_agencity
import time

def run():
    xi = np.linspace(0, 100, 5000)
    u = np.sin(xi)

    t0 = time.time()
    result = compute_agencity(xi, u)
    dt = time.time() - t0

    print("Sine benchmark time:", dt)
    return dt

if __name__ == "__main__":
    run()
