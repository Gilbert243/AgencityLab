import numpy as np
from agencitylab.api import pipeline

t = np.linspace(0, 20, 500)
u = np.sin(t) + 0.1 * np.sin(5 * t)
res = pipeline().from_arrays(t, u).compute().run()
print(res.summary())
