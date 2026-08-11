# AgencityLab overview

AgencityLab has two goals that must remain distinct:

1. provide a faithful, inspectable numerical reference implementation of the accepted Theory of Agencity;
2. provide an experimental laboratory for diagnostics, extensions, numerical studies, and empirical validation or falsification.

The first goal requires strict theory-to-code traceability. The second permits exploration, but exploratory choices must never silently replace canonical equations.

## Software layers

### `agencitylab.core`

Deterministic mathematical operators for normalization, reduced derivatives, causal moving correlation, memory/organisation, dynamic and structural intensities, logarithmic contrast, structural orientation, intrinsic state `beta`, characteristic power, and final flux `b`.

Canonical operators are separate from diagnostic or experimental helpers. Numerical safeguards may detect machine-level problems or select a safer equivalent algorithm, but they do not alter valid physical denominators or zero conditions.

### `agencitylab.api`

Stable user-facing orchestration. `compute_agencity()` is the reference scalar-signal entry point. Batch, retained-history streaming, multiscale, discrete, multivariate, export, visualization, workflow, and pipeline helpers orchestrate the same documented contracts rather than defining alternate scalar physics.

### `agencitylab.analysis`

Higher-level interpretation: coherence, angular variance, real-agencity diagnostics, regimes, events, geometry, signatures, transitions, reports, and related analysis. These consume computed results and do not redefine the canonical observable.

### `agencitylab.models`

Result and metadata containers preserve physical/contextual parameters, units, intermediate fields, backend information, producing software version, and serialization information for reproducibility.

### `tests`

The repository-level suite covers software contracts, analytical canonical identities, numerical safeguards, deterministic scientific references, diagnostics, extensions, public workflows, packaging, and compatibility.

## Canonical scalar computation

The stable scalar pipeline is:

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
Theta = atan2(O, M)
U = (M + iO) / S when S > 0, otherwise 0
beta = J U when S > 0, otherwise 0
b(t) = P_c(t) beta(t)
```

The canonical path guarantees that:

- `A_ref`, `tau`, `w`, and `P_c` are not silently inferred from ordinary signal statistics;
- `w` is the CRM memory width and is distinct from `tau` in the accepted Volume 2 formulation;
- omitting `w` uses the software convention `w=tau`, but an explicit positive `w` is preserved;
- historical `tanh` saturation does not enter `M` or `O`;
- historical `tau / A_fact` window compression does not enter canonical CRM;
- clipping, smoothing, and saturation are not injected into canonical operators;
- the logarithmic constant is the theoretical `e = exp(1)`, not numerical epsilon;
- `S = 0` is handled by an explicit branch rather than an epsilon denominator;
- arbitrarily small but non-zero quantities are not globally reclassified as physical zero.

## Stable v1.0 software contract

Version `1.0.0` stabilizes the documented user-facing interfaces without changing the equations. The principal contract includes canonical compute, result/metadata models, diagnostics, batch, retained-history streaming, multiscale, discrete, multivariate, exports, visualization, scientific workflows, and fluent orchestration.

Stable APIs follow Semantic Versioning. Experimental, research, speculative, and legacy-compatibility interfaces remain explicitly outside the strict stable guarantee. See `docs/stable_api.md` for the complete boundary.

## Exact null/rest-state convention

Following the accepted project convention, an exactly constant sampled observable is treated as the exact null/rest state before derivative or CRM evaluation:

```text
X* = A* = M = O = D = S = J = 0
U = beta = b = 0
```

No tolerance is used for this branch. Near-constant or tiny-amplitude non-constant signals still run through the canonical operators.

## Physical parameters and numerical scales

The following concepts are distinct:

- `A_ref`: physical/contextual reference amplitude, sharing the observable unit label;
- `tau`: characteristic structural time, sharing the coordinate-unit label;
- `w`: CRM memory window, also sharing the coordinate-unit label but not universally identical to `tau`;
- sampling interval: numerical discretisation of the observable;
- `P_c`: physical/contextual characteristic power, scalar or an explicitly supplied positive profile `P_c(t)`;
- `b`: observable agencity flux;
- multiscale analysis: explicit computation across selected scales rather than automatic inference of a physical `tau`.

Signal-derived estimators may exist only as clearly labelled research, diagnostic, preprocessing, or heuristic tools. They do not silently populate the stable physical parameter contract.

## Canonical versus diagnostic and experimental status

Use these labels consistently:

- **canonical**: directly implements the accepted theory definition;
- **mathematical consequence**: follows from canonical equations without a new physical assumption;
- **numerical approximation**: discrete implementation of a continuous canonical operator;
- **implementation convention**: explicit software rule such as the omission convention `w=tau`;
- **diagnostic**: interpretation quantity that does not define Agencity;
- **experimental**: deliberate alternative or extension under investigation;
- **research/speculative**: theoretical extensions not established as part of the stable scalar implementation;
- **legacy**: historical compatibility behaviour outside the canonical reference path.

`beta != 0` is not a definition of real agencity. Real-agencity analysis separately evaluates structural validity, angular stability/coherence, and contextual significance of `|b|`.

## Scientific limits

Version 1.0 establishes a stable software contract. It does not establish universal empirical validity, universal regime thresholds, or experimental confirmation of research extensions. Results remain conditional on the observable, sampling/preprocessing choices, and physically/contextually justified parameters. The inverse problem is non-injective, current streaming is retained-history recomputation rather than O(1)-memory recurrence, and accelerated backends remain experimental.
