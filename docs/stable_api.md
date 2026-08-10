# Stable computational API — v0.3

Version 0.3 stabilizes the user-facing scalar-signal computational interface without changing the canonical equations introduced in v0.2.

## Reference call

```python
import numpy as np
from agencitylab import compute_agencity

xi = np.arange(8.0)
u = np.sin(xi)

result = compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    P_c=5.0,
    unit="rad",
    coordinate_unit="s",
    power_unit="W",
)
```

`u` and `xi` must be finite one-dimensional arrays with the same length and at least three samples. `xi` must be strictly increasing. The compatibility alias `data=` remains accepted, but `u=` and `data=` cannot be supplied together.

## Physical parameters

`A_ref`, `tau`, and `P_c` remain physical/contextual inputs. They are never silently inferred from signal statistics.

`A_ref` and `tau` may be passed explicitly, stored in `ExperimentMetadata`, or resolved by a deliberately registered physical convention. A scalar `P_c` may follow the same routes.

The canonical observable is `b(t) = P_c(t) beta(t)`. Therefore the stable API also accepts an explicitly supplied time-varying `P_c(t)` as either:

```python
P_c = np.linspace(10.0, 12.0, len(xi))
result = compute_agencity(..., P_c=P_c)
```

or:

```python
result = compute_agencity(..., P_c=lambda t: 10.0 + 0.2 * t)
```

A sampled power profile must be finite, strictly positive, one-dimensional, and match `xi`. It is contextual input; AgencityLab does not derive it from `u`. `ExperimentMetadata.characteristic_power` remains the scalar metadata field; a time-varying profile is preserved on `AgencityResult.P_c` and labelled as time-varying in metadata `extra`.

The canonical CRM window is `w = tau`. Passing a different `w` is an explicit error.

## Units

AgencityLab 0.3 records unit labels; it does not perform hidden unit conversion.

- `unit` labels `u` and `A_ref`.
- `coordinate_unit` labels `xi` and `tau`.
- `power_unit` labels `P_c`.
- `b` carries the corresponding informational-power label `power_unit·nat`; for `power_unit="W"`, `result.b_unit == "W·nat"`.

If `xi` is omitted, the API generates a sample index and records `coordinate_unit="sample"` unless the caller supplied another explicit label.

`ExperimentMetadata.unit_contract()` returns the unit-label mapping used by a result.

## Result model

`AgencityResult` is the stable container returned by `compute_agencity`. It validates array lengths, finiteness, strictly positive physical scales, power-profile shape, and metadata consistency.

The result exposes the canonical intermediate fields:

```text
xi, u, u_star, X_star, A_star, t_star,
M, O, D, S, J, U, theta, beta, P_c, b
```

`theta` is the canonical wrapped structural orientation represented by `angle(U)`. Phase unwrapping is an analysis operation and is not performed by the result model.

`result.to_dict()` emits schema version `0.3`; `AgencityResult.from_dict()` restores complex arrays, scalar or sampled `P_c`, and retains compatibility aliases for older serialized physical-field names. When older payloads stored the physical scales only in metadata, v0.3 resolves them from that metadata before using any compatibility default.

Legacy summary keys such as `Pc_mean`, `A_fact`, and `resolution_scale` remain available. New fields indicate whether `P_c` is time-varying.

## Explicit errors

Applications may catch the public exception hierarchy:

```python
from agencitylab import (
    AgencityError,
    AgencityValidationError,
    PhysicalParameterError,
    UnitValidationError,
    BatchItemError,
    StreamStateError,
    StreamNotReadyError,
)
```

Validation errors remain subclasses of `ValueError`, preserving compatibility with code that previously caught `ValueError`, while providing more specific types for applications.

Unknown `compute_agencity` keyword arguments are rejected instead of being silently stored or ignored. Legacy `A_fact` / `activity_factor` cannot modify canonical CRM, and `resolution_scale` cannot silently insert preprocessing into the canonical computation.

## Batch computation

`run_batch()` accepts raw signals, `(xi, u)` tuples, or item dictionaries. Item dictionaries may carry per-item physical parameters, metadata, config, or presets. Per-item values override batch-wide values, results preserve input order, and failures identify the zero-based item index through `BatchItemError`.

```python
from agencitylab import run_batch

results = run_batch(
    [
        {"xi": xi, "u": u, "P_c": 1.0},
        {"xi": xi, "u": u, "P_c": 2.0},
    ],
    A_ref=1.0,
    tau=2.0,
)
```

## Streaming

`AgencityStream` can retain physical parameters across updates. When coordinates are omitted, generated sample coordinates continue monotonically across chunks rather than restarting from zero.

```python
from agencitylab import AgencityStream

stream = AgencityStream(
    analyze=False,
    A_ref=1.0,
    tau=1.0,
    P_c=1.0,
)

result = stream.update([0.0, 1.0, 0.0, -1.0])
result = stream.update([0.0, 1.0, 0.0, -1.0])
```

Explicit coordinate chunks must be finite, strictly increasing, and start after the previous coordinate. If the buffer does not yet contain enough history for two CRM windows, the stream raises `StreamNotReadyError` while retaining the buffered samples.

For streaming, a persistent scalar or callable `P_c` is generally the clearest contract. A sampled profile may also be supplied when its length matches the current rolling computation buffer; mismatches are explicit errors.

## Fluent pipeline compatibility

`pipeline().set_tau(...)` and `set_power(...)` update the actual physical metadata used by computation. `set_resolution_scale(...)` remains observational metadata only and does not inject smoothing into canonical equations. `set_activity_factor(...)` is retained as a deprecated metadata compatibility method and does not alter canonical CRM.

## Scientific boundary

Version 0.3 is an API-stability milestone. It does not change `beta`, `J`, CRM, `M`, `O`, `D`, `S`, `tau`, `w`, `P_c`, or `A_ref` merely for software convenience, and it does not constitute empirical validation of the Theory of Agencity.

Supporting an externally specified `P_c(t)` is an API realization of the canonical relation `b(t) = P_c(t) beta(t)`, not a signal-derived extension.
