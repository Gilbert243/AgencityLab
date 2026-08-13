# Public API contract — AgencityLab 1.0

AgencityLab 1.0 establishes the first stable public software contract around the
accepted Theory of Agencity. Stability applies to documented user-facing
interfaces and their stated semantics; it does not redefine the theory and does
not convert software validation into empirical confirmation.

The **sole reference canonical scalar orchestration** is:

```text
agencitylab.compute_agencity()
```

with the accepted chain:

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,U,Theta -> beta -> b
```

## Stable 1.0 entry points

The principal stable contract includes:

- `compute_agencity()` — reference canonical scalar computation;
- `AgencityResult` and `ExperimentMetadata`;
- `analyze_agencity()` and named diagnostic helpers;
- `run_batch()` / `analyze_batch()`;
- `AgencityStream` / `stream_agencity()`;
- `compute_agencity_spectrum()`;
- `compute_discrete_agencity()` — explicit Volume-2 discrete construction;
- `compute_multivariate_agencity()`;
- documented JSON/CSV and optional Excel/PDF exports;
- documented visualisation helpers;
- `ScientificStudy` / `scientific_workflow()`;
- `AgencityPipeline` / `pipeline()`.

Specialized functionality is reached through its owning namespace instead of a
large package-root alias table.

## Semantic Versioning policy

Starting with 1.0.0:

- patch releases (`1.0.x`) are compatible bug fixes, documentation and internal improvements;
- minor releases (`1.x.0`) may add backwards-compatible functionality or APIs;
- major releases (`2.0.0`+) are required for intentional breaking changes to the stable public contract.

Ordinary public removals after 1.0 follow the lifecycle documented in
`SUPPORT.md`. A correctness, security, or scientifically invalid interface may
be removed sooner when retaining it would mislead users; such a correction must
be explicit in release notes.

Repository snapshots preceding 1.0 are development history and do not create
stable compatibility obligations.

## Canonical physical-parameter contract

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual quantities and are not
silently estimated from the observed signal.

- `A_ref > 0` is the fixed reference amplitude used exactly by `u* = u/A_ref`.
  Standard deviation, variance, MAD, range, min/max and z-score are not canonical fallbacks.
- `tau > 0` is characteristic structural time.
- `w > 0` is the CRM memory width and is distinct from `tau`. If `w` is omitted,
  AgencityLab applies the implementation fallback `w=tau`; this is not a universal identity.
- `P_c >= 0` is finite characteristic physical/contextual power. It may be a
  scalar, sampled profile matching `xi`, or supported callable. Exact zero is valid and gives `b=0`.

When `w` is omitted, results record:

```text
w was unspecified; implementation convention w = tau was used
```

Sampling interval, `tau`, CRM width `w`, and multiscale scanning are distinct objects.

## Canonical identities

For the reference scalar result:

```text
S = sqrt(M^2 + O^2)
Theta = atan2(O, M)
J = ln((e + D) / (e + S)),  e = exp(1)
```

For `S > 0`:

```text
U = (M + i O) / S
|U| = 1
beta = J U
b = P_c beta
```

For `S = 0`:

```text
U = 0
beta = 0
```

`S=0`, `beta=0`, or `P_c=0` are valid physical states, not numerical anomalies.
No epsilon is inserted into these equations.

## Reference call

```python
import numpy as np
from agencitylab import compute_agencity

xi = np.linspace(0.0, 20.0, 801)
u = np.sin(xi)

result = compute_agencity(
    u,
    xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
    coordinate_unit="s",
    power_unit="W",
)
```

`u` and `xi` must be finite one-dimensional arrays of equal length with at least
three samples. `xi` must be strictly increasing.

## Result and reproducibility contract

`AgencityResult` exposes canonical data:

```text
xi, u, u_star, X_star, A_star, t_star,
M, O, D, S, J, U, theta, beta, P_c, b
```

It also preserves physical parameters, unit/context labels, producing
AgencityLab version and reproducibility metadata. Diagnostic analyses, reports,
signatures and multiscale products are workflow artifacts, not mutable fields
on the canonical result model.

Result serialization uses schema `1.0`. Complex `beta` and `b` are preserved by
JSON serialization; CSV uses explicit real/imaginary/magnitude columns. The 1.0
deserializer is strict and does not silently reconstruct development-only schemas.

When `P_c=0`, the canonical flux remains exactly zero. The convenience inverse
ratio `eta=|b|/P_c` is mathematically undefined at those samples and is represented
as `NaN`; no epsilon or artificial reconstruction of `beta` is used.

## Continuous sampled computation versus discrete Volume-2 computation

`compute_agencity()` is the reference continuous-theory orchestration. Its
derivatives are numerically sampled with the ordinary finite-difference
operators in `activation.py` and `activity.py`, conceptually:

```text
X* ~= gradient(u*)
A* ~= gradient(X*)
```

`compute_discrete_agencity()` has a different, explicit contract. It implements
the Volume-2 interior stencils:

```text
X_n = (u[n+1] - u[n-1]) / (2 delta)
A_n = (u[n+1] - 2u[n] + u[n-1]) / delta^2
```

on the canonical reduced sequence `u_star` with `delta_star=delta/tau`. The
direct Volume-2 second difference is not silently replaced with
`gradient(gradient(u))`.

## Diagnostics are not canonical physics

`agencitylab.analysis` is the reference interpretation layer. Coherence,
angular variance, real-agencity, curvature, winding, events, transitions,
signatures, regimes and reports consume canonical results and never modify
them.

`beta != 0` is not a definition of coherent or real agencity. Contextual
diagnostics may combine structural validity `S>0`, angular stability /
`Sigma_Theta`, significant `|b|`, and persistence. Thresholds are diagnostic
inputs, not universal constants.

## Batch

`run_batch()` accepts supported raw signals, `(xi,u)` tuples, or item
dictionaries. Items may carry independent `A_ref`, `tau`, `w`, `P_c` and
metadata. Results preserve input order, and supported parallel execution must
not alter scientific results beyond documented numerical tolerances.

## Streaming

`AgencityStream` retains history and recomputes the retained record. With full
history, its final result is expected to match one-shot computation. A finite
`window_size` intentionally defines a different retained-history problem.
Version 1.0 does not claim an O(1)-memory online recurrence.

## Multiscale

`compute_agencity_spectrum()` scans explicit `tau` values. With `windows=None`,
each row applies the implementation fallback `w=tau`; `windows=` may provide
independent widths. This is not automatic estimation of physical `tau` and does
not conflate `tau`, `w`, or sampling interval.

## Multivariate computation and zero power

`compute_multivariate_agencity()` computes scalar states by component and the
vector-additive total flux `sum_k P_c,k beta_k`. `P_c,k=0` is valid. Where total
component power is zero, the weighted mean `beta_multi` is mathematically
undefined; the stable array representation stores zero and exposes
`beta_multi_defined=False` explicitly rather than adding epsilon.

## Runtime and acceleration boundary

The canonical 1.0 scalar pipeline is NumPy based. `agencitylab.config` controls
software/runtime behaviour only; it does not supply physical quantities.
Experimental Numba and JAX primitives remain under `agencitylab.backends` and
do not masquerade as a complete alternate canonical pipeline.

## Experimental and research interfaces

The following remain outside the stable canonical contract:

- Numba backend primitives;
- JAX backend primitives and device/precision behaviour;
- signal-derived `optimize_agencity_window()` as an explicit window-selection study;
- incomplete Riemannian extension;
- GPU, distributed and out-of-core execution;
- true constant-memory online streaming recurrence;
- field, extended thermodynamic, quantum, gravitational and cosmological research/speculative modules.

NumPy remains the stable complete canonical backend.

## Scientific limitations

The stability guarantee is a software contract, not empirical confirmation of
the Theory of Agencity. Results remain conditional on the chosen observable and
physically/contextually justified parameters. Sampling and preprocessing
assumptions must be explicit. Sensitivity to `w` is scientifically meaningful.
The inverse problem is non-injective. Accelerated backends and fundamental
extensions remain separately labelled experimental/research/speculative.
