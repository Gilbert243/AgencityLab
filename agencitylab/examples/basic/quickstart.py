import numpy as np
from agencitylab.api import compute_agencity, visualize_agencity, analyze_agencity

t = np.linspace(0, 10, 200)
u = np.sin(t)
result = compute_agencity(t, u)
print(result.summary())
visualize_agencity(result)
print(analyze_agencity(result))
