import numpy as np
from agencitylab.api import compute_agencity, visualize_agencity, analyze_agencity

if __name__ == "__main__":
    t = np.linspace(0, 10, 200)
    u = np.sin(t)
    result = compute_agencity(t, u)
    print(result.summary())
    print(analyze_agencity(result))
    visualize_agencity(result)
