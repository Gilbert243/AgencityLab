# Basic Usage

A minimal stable v1.0 computation uses explicit physical/contextual parameters:

```python
import numpy as np
import agencitylab

# A 0.02 s sampling interval makes tau=1.0 s and w=0.8 s exact window multiples.
xi = np.linspace(0.0, 6.0, 301)
u = np.sin(xi)

result = agencitylab.compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=1.0,
    w=0.8,
    P_c=2.0,
    coordinate_unit="s",
    power_unit="W",
)

print(result.b)
```

`A_ref`, `tau`, `w`, and `P_c` are not silently inferred from the signal. If `w` is omitted, the stable API uses the documented software convention `w=tau`. For uniformly sampled data, a CRM window must also be representable by an integer number of sampling intervals.

For diagnostics:

```python
analysis = agencitylab.analyze_agencity(result)
print(analysis["real_agencity"]["status"])
```

This same public compute/analyze contract is exercised by the release test suite and built-wheel end-to-end CI gate.
