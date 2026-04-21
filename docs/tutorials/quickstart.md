# Quickstart

```python
import numpy as np
from agencitylab.api import compute_agencity

xi = np.linspace(0, 10, 200)
u = np.sin(xi)

result = compute_agencity(xi, u)
print(result.summary())
```
