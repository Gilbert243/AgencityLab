# AgencityLab overview

AgencityLab has two goals that must remain distinct:

1. provide a faithful, inspectable numerical reference implementation of the current Theory of Agencity;
2. provide an experimental laboratory for diagnostics, alternative formulations, multiscale studies, and future validation.

The first goal requires strict theory-to-code traceability. The second allows exploration, but exploratory choices must never silently replace canonical equations.

## Software layers

### `agencitylab.core`

Deterministic mathematical operators for normalization, reduced derivatives, causal moving correlation, memory/organisation, dynamic and structural intensities, logarithmic contrast, structural orientation, intrinsic state `beta`, characteristic power, and final flux `b`.

Canonical operators are kept separate from diagnostic or experimental helpers. Generic numerical safeguard functions may exist in the package, but canonical equations do not use machine epsilon to alter valid physical denominators or zero conditions.

### `agencitylab.api`

User-facing orchestration. `compute_agencity` is the reference scalar-signal entry point and follows the current second-edition pipeline without duplicating alternative equations. Version 0.3 stabilizes input validation, error types, unit labels, batch execution, streaming, and fluent-pipeline compatibility around that core.

### `agencitylab.analysis`

Higher-level interpretation: coherence, regimes, information measures, events, geometry, signatures, anomalies, and multiscale tools. These are diagnostics or experimental analyses and do not redefine the canonical observable.

### `agencitylab.models`

Result and metadata containers preserve physical/contextual parameters, units, intermediate fields, and serialization information for reproducibility. `AgencityResult` uses serialization schema `0.3`.

### `tests`

The repository-level test suite contains software-foundation checks, analytical tests of each canonical operator, end-to-end identities, and stable public-API tests.

## Canonical guarantees inherited from 0.2.0

The reference scalar-signal computation remains:

```text
u* = u / A_ref
t* = t / tau
X* = d(u*) / d(t*)
A* = d^2(u*) / d(t*)^2
M = CRM[u*]
O = CRM[u*, X*]
D = sqrt((X*)^2 + (A* X*)^2)
S = sqrt(M^2 + O^2)
J = ln((e + D) / (e + S))
U = (M + iO) / S when S > 0, otherwise 0
beta = J U when S > 0, otherwise 0
b(t) = P_c(t) beta(t)
```

The canonical path also guarantees that:

- `A_ref`, `tau`, and `P_c` are not silently inferred from the observed signal;
- the canonical CRM window is `w = tau`;
- historical `tanh` saturation does not enter `M` or `O`;
- historical `tau / A_fact` window compression does not enter canonical CRM;
- clipping, smoothing, and saturation are not injected into canonical operators;
- the logarithmic constant is the theoretical `e = exp(1)`, not a numerical epsilon;
- `S = 0` is handled by an explicit branch rather than an epsilon denominator;
- arbitrarily small but non-zero values are not globally reclassified as zero.

## Stable computational API guarantees for 0.3.0

Version `0.3.0` adds software guarantees without changing those equations:

- `compute_agencity()` accepts one finite one-dimensional scalar observable and a strictly increasing coordinate;
- `data=` remains an alias for `u=`, but ambiguous dual input is rejected;
- unknown compute keywords are rejected rather than silently ignored;
- public exceptions distinguish validation, physical-parameter, unit, batch, and streaming failures;
- `ExperimentMetadata` records unit labels and preserves unknown fields for forward compatibility;
- `AgencityResult` validates array lengths and physical scales, supports scalar or externally supplied sampled `P_c(t)`, and serializes with schema version `0.3`;
- canonical `theta` remains wrapped `atan2(O, M)` / `angle(U)`; phase unwrapping belongs to analysis;
- unit labels are descriptive only and never trigger hidden conversions;
- `P_c` is labelled in power units and `b` in the corresponding informational-power unit, e.g. `W·nat`;
- batch execution supports per-item physical context and identifies failing item indices;
- implicit streaming coordinates continue across chunks instead of restarting at zero;
- streaming reports insufficient CRM history explicitly rather than exposing an opaque low-level failure;
- fluent-pipeline compatibility setters now modify the physical metadata actually consumed by computation.

See `docs/stable_api.md` for the detailed contract.

## Exact null/rest-state postulate

Following the current project maintainer interpretation, exact rest is a canonical postulate for implementation rather than a numerical theorem to be established by finite differences.

If the sampled observable is exactly constant, the reference pipeline checks this condition before derivative or CRM evaluation and returns

```text
X* = A* = M = O = D = S = J = 0
U = beta = b = 0
```

exactly. No tolerance is used. Near-constant or tiny-amplitude non-constant signals still run through the canonical operators.

## Physical parameters, units, and numerical scales

The following concepts must remain separate:

- `A_ref`: physical/contextual reference amplitude, with the same unit label as `u`;
- `tau`: intrinsic characteristic structural time, with the same coordinate-unit label as `xi`;
- `w`: CRM memory window, fixed to `tau` in the canonical path;
- sampling interval: numerical discretisation of the observable;
- `P_c`: physical characteristic power, normally scalar for a fixed container but representable as an explicitly supplied profile `P_c(t)` when the physical context is time-dependent;
- `b`: observable informational-power flux, labelled `power_unit·nat`;
- multiscale analysis: diagnostic/experimental analysis across scales.

Signal-derived estimators for `tau`, `P_c`, or normalization scales may remain available for research workflows only when explicitly labelled experimental or heuristic. AgencityLab 0.3 records unit labels but does not convert magnitudes between unit systems.

## Canonical versus experimental status

Use the following labels in code review and documentation:

- **canonical**: directly implements the current selected theory definition;
- **mathematical consequence**: follows from canonical equations without a new physical assumption;
- **numerical approximation**: discrete implementation of a continuous canonical operator;
- **implementation convention**: explicit software rule needed to realize the theory numerically;
- **experimental**: deliberate alternative or extension under investigation;
- **heuristic**: practical modelling rule not promoted to a theoretical axiom;
- **diagnostic**: interpretation quantity that does not define Agencity;
- **legacy**: historical behaviour retained outside the canonical reference path.

## What 0.3.0 does not claim

Version 0.3.0 establishes a more usable and explicit computational interface. It does not establish empirical validity across domains, universal agency classification, or experimental confirmation of field-theory, quantum, gravitational, or cosmological extensions.

Future phases can test simple dynamical systems, coherence diagnostics, scaling limits, stochastic processes, chaos, multiscale behaviour, and empirical falsification without changing the canonical equations merely to obtain preferred results.
