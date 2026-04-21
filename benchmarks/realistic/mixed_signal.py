import numpy as np
from agencitylab.api.compute import compute_agencity
import time

def run():
    xi = np.linspace(0, 50, 3000)
    u = np.sin(xi) + 0.5*np.cos(3*xi) + 0.1*np.random.randn(len(xi))

    t0 = time.time()
    compute_agencity(xi, u)
    dt = time.time() - t0

    print("Mixed signal benchmark:", dt)
    return dt

if __name__ == "__main__":
    run()
