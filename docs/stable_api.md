# Public API contract — v1.0 Stable Scientific Release

AgencityLab 1.0 establishes the first stable public software contract around the accepted Theory of Agencity. Stability applies to documented user-facing interfaces and their stated semantics; it does not redefine the theory and does not convert software validation into empirical confirmation.

The stable orchestration layer implements the accepted canonical scalar pipeline:

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,U,Theta -> beta -> b
```

## Stable v1.0 entry points

The following interfaces form the principal v1.0 stable contract and should evolve according to Semantic Versioning:

- `compute_agencity()` — canonical scalar computation.
- `AgencityResult` and `ExperimentMetadata` — result and reproducibility models.
- `analyze_agencity()` and the named `analyze_*` diagnostics — analysis of an already-computed result; diagnostics never modify canonical arrays.
- `run_batch()` / `analyze_batch()` — ordered multi-signal orchestration with per-item physical context.
- `AgencityStream` / `stream_agencity()` — retained-history streaming orchestration with the documented recomputation semantics.
- `compute_agencity_spectrum()` — explicit multiscale computation.
- `compute_discrete_agencity()` and `compute_multivariate_agencity()` — theory-defined sampled and multivariate constructions.
- `export_json()`, `export_csv()`, `export_result_csv()`, `export_study_json()` and the optional Excel/PDF exporters.
- `visualize_agencity()` / `visualize_multiscale_spectrum()` — optional presentation of computed results.
- `ScientificStudy` / `scientific_workflow()` — convenience orchestration; they do not replace the canonical engine.
- `AgencityPipeline` / `pipeline()` — fluent orchestration of the same public computation.

Top-level convenience helpers such as `run`, `inspect`, `plot`, `quick_summary`, `generate_report`, `textual_analysis`, and batch summaries remain supported conveniences around these stable objects. They are not alternate definitions of the theory.

## Semantic Versioning policy

Starting with 1.0.0:

- patch releases (`1.0.x`) are for bug fixes, documentation, and compatible internal improvements;
- minor releases (`1.x.0`) may add backwards-compatible functionality or new public APIs;
- major releases (`2.0.0` and later) are required for intentional breaking changes to the stable public contract.

A scientifically necessary correction can require a breaking change if the accepted theory and implementation are demonstrably inconsistent. Such a correction must be documented explicitly rather than hidden behind compatibility behaviour.

Interfaces explicitly labelled experimental, research, speculative, or legacy compatibility may evolve more freely and are not automatically covered by the stable API guarantee.

## Experimental and research interfaces

The following remain explicitly outside the stable v1.0 contract:

- Numba backend primitives;
- JAX backend primitives and their device/precision behaviour;
- `optimize_agencity_window()`, which performs an explicit signal-derived Chapter 13 window-selection study and must not be confused with a physically specified `w` or with `tau`;
- the Riemannian extension, for which `riemannian_extension_status()` reports that a production pipeline is not implemented;
- GPU, distributed, and out-of-core execution;
- a true constant-memory online streaming recurrence;
- field, extended thermodynamic, quantum, gravitational, and cosmological modules that are research/speculative extensions rather than part of the stable scalar reference contract.

NumPy remains the stable complete canonical backend. Requesting Numba or JAX records an experimental primitive-backend request but does not silently replace the complete NumPy reference pipeline.

## Compatibility and legacy

Compatibility behaviour is deliberately isolated:

- `data=` remains an alias for `u=`; supplying both is an error.
- `Pc=` remains a legacy spelling for `P_c=` in `compute_agencity`; new code should use `P_c`.
- `PipelineBuilder` and `pipeline_builder` remain compatibility aliases for `AgencityPipeline` and `pipeline`.
- legacy result fields such as `A_fact`, `resolution_scale`, and older serialized physical-field names remain readable where documented, but they do not alter the canonical equations.
- signal-derived `estimate_tau()` and statistical normalization helpers are explicitly experimental/preprocessing utilities, not canonical physical-parameter inference.

Historical `tanh` saturation, `tau/A_fact` CRM compression, epsilon-modified physical denominators, and silent signal-statistical fallbacks are not part of the reference pipeline.

## Reference call

```python
import numpy as np
from agencitylab import compute_agencity

xi = np.linspace(0.0, 20.0, 801)
u = np.sin(xi)

result = compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
    unit="rad",
    coordinate_unit="s",
    power_unit="W",
)
```

`u` and `xi` must be finite one-dimensional arrays of equal length with at least three samples. `xi` must be strictly increasing.

## Physical parameter contract

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual quantities and are not silently estimated from the observed signal.

- `A_ref > 0` is the reference amplitude used exactly by `u* = u / A_ref`.
- `tau > 0` is the characteristic structural time.
- `w > 0` is the CRM memory width. Volume 2 keeps `w` distinct from `tau`; omitting `w` uses the software convention `w=tau`, while an explicit positive `w` is preserved. This omission convention is not a universal identity between the two quantities.
- `P_c > 0` may be a scalar, a sampled positive profile matching `xi`, or a callable evaluated on `xi`. It is external/contextual and is never inferred from `u`.

Unit labels are descriptive only. `unit` applies to `u` and `A_ref`; `coordinate_unit` applies to `xi`, `tau`, and `w`; `power_unit` applies to `P_c`. No hidden conversion is performed.

## Canonical identities

For the canonical result:

```text
S = sqrt(M^2 + O^2)
Theta = atan2(O, M)
J = ln((e + D) / (e + S))
```

For `S > 0`:

```text
U = (M + i O) / S
|U| = 1
beta = J U
b = P_c beta
```

For `S = 0`, the canonical convention is `U = 0` and `beta = 0`. Numerical epsilon is not inserted into these valid equations.

## Result and reproducibility contract

`AgencityResult` exposes the canonical computational state:

```text
xi, u, u_star, X_star, A_star, t_star,
M, O, D, S, J, U, theta, beta, P_c, b
```

`theta` is the wrapped canonical orientation. Unwrapping belongs to analysis.

The result preserves or exposes:

- the input coordinate and observable;
- `A_ref`, `tau`, `w`, and `P_c`;
- unit/context labels;
- backend request, resolved backend, backend status, and canonical backend in configuration;
- the AgencityLab version that produced a new computation;
- user/domain metadata;
- canonical intermediate arrays needed for inspection and reproducibility.

Complex `beta` and `b` are preserved by the JSON serialization contract. CSV exports use explicit real/imaginary/magnitude columns and do not silently discard the imaginary component.

This metadata supports traceability of a computation. It does not make the inverse problem injective and does not imply that the original observable can be reconstructed uniquely from `b` alone.

## Validation and errors

The stable API rejects empty/too-short or non-finite signals, non-numeric or multidimensional scalar inputs, inconsistent axes, non-increasing coordinates, invalid physical parameters, incompatible power-profile shapes, contradictory aliases, and unknown compute keywords before they can become cryptic NumPy failures.

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

## Batch

`run_batch()` accepts raw signals, `(xi, u)` tuples, or item dictionaries. Item dictionaries may carry independent `A_ref`, `tau`, `w`, `P_c`, metadata and configuration. Results preserve input order, and supported parallel execution must not alter the scientific result beyond documented numerical tolerances.

## Streaming

`AgencityStream` retains history and recomputes the retained record after updates. With full history, its final result is expected to match one-shot computation. An explicit finite `window_size` intentionally limits retained history and therefore defines a different retained-history problem. Version 1.0 does not claim an O(1)-memory online recurrence.

## Multiscale

`compute_agencity_spectrum()` scans explicit `tau` values. By default each row uses the software convention `w=tau`; `windows=` may supply independent positive widths. A multiscale scan is not automatic estimation of the physical `tau` and does not conflate `tau`, `w`, or the sampling interval.

## Discrete and multivariate computation

`compute_discrete_agencity()` is a sampled-sequence convenience around the accepted discrete construction. Its boundary conventions and dimensional contract are covered by the extension tests.

`compute_multivariate_agencity()` computes the accepted per-component scalar states with component context and applies the documented multivariate aggregation. It does not promote the incomplete Riemannian research extension into the stable API.

## Diagnostics are not canonical physics

`agencitylab.analysis` consumes computed results. Coherence, angular variance, curvature, winding, events, transitions, signatures, regimes, reports, and real-agencity assessments remain a diagnostic layer.

In particular, `beta != 0` is not a definition of coherent or “real” agencity. Real-agencity diagnostics combine structural validity (`S > 0`), contextual angular stability/coherence, and a contextual significance condition on `|b|`. Thresholds and persistence rules are diagnostic inputs, not universal constants and never modify the canonical `beta` function.

## Scientific limitations

The 1.0 stability guarantee is a software contract, not an empirical validation claim. Results remain conditional on the chosen observable and physically/contextually justified parameters. Preprocessing and sampling assumptions must be explicit. Sensitivity to CRM window `w` is scientifically meaningful rather than something to hide behind automatic parameter inference. Research and speculative extensions remain separately labelled.
