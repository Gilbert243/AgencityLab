# Full Pipeline Tutorial

Use the chainable pipeline when you want a reproducible workflow:

```python
from agencitylab.api import pipeline

context = (
    pipeline()
    .from_arrays(xi, u)
    .preprocess(normalize=True, smooth=True)
    .compute()
    .analyze()
    .run()
)
```
