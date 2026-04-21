# Pipeline API

The pipeline API provides a chainable interface for reproducible workflows.

Example:

```python
from agencitylab.api import pipeline

result = (
    pipeline()
    .from_arrays(xi, u)
    .compute()
    .analyze()
    .run()
)
```
