import numpy as np
def solve_field(initial, steps=10):
    arr = np.asarray(initial, dtype=float)
    return np.stack([arr for _ in range(steps)], axis=0)
