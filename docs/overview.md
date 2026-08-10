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

User-facing orchestration. `compute_agencity` is the reference scalar canonical entry point and follows the current second-edition pipeline without duplicating alternative equations.

### `agencitylab.analysis`

Higher-level interpretation: coherence, regimes, information measures, events, geometry, signatures, anomalies, and multiscale tools. These are diagnostics or experimental analyses and do not redefine the canonical observable.

### `agencitylab.models`

Result and metadata containers that preserve physical/contextual parameters and intermediate fields for reproducibility.

### `tests`

The repository-level test suite contains software-foundation checks plus analytical tests of each canonical operator and the end-to-end identities.

## Canonical guarantees for 0.2.0

Version `0.2.0` guarantees that the reference scalar computation uses:

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
b = P_c beta
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

## Exact null/rest-state postulate

Following the current project maintainer interpretation, exact rest is a canonical postulate for implementation rather than a numerical theorem to be established by finite differences.

If the sampled observable is exactly constant, the reference pipeline checks this condition before derivative or CRM evaluation and returns

```text
X* = A* = M = O = D = S = J = 0
U = beta = b = 0
```

exactly. No tolerance is used. Near-constant or tiny-amplitude non-constant signals still run through the canonical operators.

## Physical parameters and numerical scales

The following concepts must remain separate:

- `A_ref`: physical/contextual reference amplitude;
- `tau`: intrinsic characteristic structural time;
- `w`: CRM memory window, fixed to `tau` in the v0.2 canonical path;
- sampling interval: numerical discretisation of the observable;
- `P_c`: physical characteristic power;
- multiscale analysis: diagnostic/experimental analysis across scales.

Signal-derived estimators for `tau`, `P_c`, or normalization scales may remain available for research workflows only when explicitly labelled experimental or heuristic.

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

## What 0.2.0 does not claim

Version 0.2.0 establishes implementation fidelity of the scalar canonical core. It does not establish empirical validity across domains, universal agency classification, or experimental confirmation of field-theory, quantum, gravitational, or cosmological extensions.

The next scientific phases can test simple dynamical systems, coherence diagnostics, scaling limits, stochastic processes, chaos, multiscale behaviour, and empirical falsification without changing the canonical equations merely to obtain preferred results.
