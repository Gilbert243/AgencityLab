import numpy as np
from agencitylab.api.compute import compute_agencity
import time

sizes = [100, 1000, 5000, 10000]

def run():
    results = {}
    for n in sizes:
        xi = np.linspace(0, 100, n)
        u = np.sin(xi)

        t0 = time.time()
        compute_agencity(xi, u)
        dt = time.time() - t0

        results[n] = dt
        print(f"Size {n}: {dt:.4f}s")
    return results

if __name__ == "__main__":
    run()
