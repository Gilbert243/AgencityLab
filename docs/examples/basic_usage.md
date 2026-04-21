# Basic Usage

A minimal example:

```python
import numpy as np
from agencitylab.api import compute_agencity

xi = np.linspace(0, 6.28, 300)
u = np.sin(xi)
result = compute_agencity(xi, u)
```
