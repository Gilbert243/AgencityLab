# Public API contract — v0.9 Release Candidate

Version 0.9 is the API-freeze candidate for AgencityLab 1.0. It does not redefine the Theory of Agencity. The stable contract is the public orchestration layer around the accepted canonical pipeline:

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,U,Theta -> beta -> b
```

## Stable candidate-v1.0 entry points

The following user-facing interfaces are treated as stable candidates for v1.0 and should only receive backwards-compatible changes unless a scientific error is discovered:

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

Top-level convenience helpers such as `run`, `inspect`, `plot`, `quick_summary`, `generate_report`, `textual_analysis`, and batch summaries are supported conveniences around these stable objects. They are not alternate definitions of the theory.

## Experimental interfaces

The following remain explicitly experimental and are not promoted merely for the Release Candidate:

- Numba backend primitives;
- JAX backend primitives and their device/precision behavior;
- `optimize_agencity_window()`, which performs an explicit signal-derived Chapter 13 window-selection study and must not be confused with a physically specified `w` or with `tau`;
- the Riemannian extension, for which `riemannian_extension_status()` reports that a production pipeline is not implemented;
- field, thermodynamic, and other speculative extension modules that are not part of the stable top-level computational contract.

NumPy remains the stable canonical backend. Requesting Numba or JAX does not silently replace the complete NumPy reference pipeline.

## Compatibility and legacy

Compatibility behavior is deliberately isolated:

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
- `w > 0` is the CRM memory width. Volume 2 keeps `w` distinct from `tau`; omitting `w` uses the common software convention `w=tau`, while an explicit positive `w` is preserved.
- `P_c > 0` may be a scalar, a sampled positive profile matching `xi`, or a callable evaluated on `xi`. It is external/contextual and is never inferred from `u`.

Unit labels are descriptive only. `unit` applies to `u` and `A_ref`; `coordinate_unit` applies to `xi`, `tau`, and `w`; `power_unit` applies to `P_c`. No hidden conversion is performed.

## Result and reproducibility contract

`AgencityResult` exposes the canonical arrays:

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

`run_batch()` accepts raw signals, `(xi, u)` tuples, or item dictionaries. Item dictionaries may carry independent `A_ref`, `tau`, `w`, `P_c`, metadata and configuration. Results preserve input order, and parallel execution must not alter numerical results.

## Streaming

`AgencityStream` retains history and recomputes the retained record after updates. With full history, its final result is expected to match one-shot computation. An explicit finite `window_size` intentionally limits retained history and therefore defines a different retained-history problem. v0.9 does not claim an O(1)-memory online recurrence.

## Multiscale

`compute_agencity_spectrum()` scans explicit `tau` values. By default each row uses the convention `w=tau`; `windows=` may supply independent positive widths. A multiscale scan is not automatic estimation of the physical `tau`.

## Scientific boundary

The stable software contract does not convert diagnostics into canonical physics. `beta != 0` does not establish coherent or real agencity, and analysis thresholds remain contextual. A Release Candidate validates implementation and usage contracts; it is not empirical confirmation of the Theory of Agencity.
