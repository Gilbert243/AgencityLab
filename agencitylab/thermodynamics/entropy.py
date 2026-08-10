import numpy as np
def agential_entropy(x):
    x = np.asarray(x, dtype=float)
    p = np.abs(x)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
