# v1.0 Stable Scientific Release readiness

This checklist is the software release gate for AgencityLab 1.0. It records implementation and distribution readiness; it does not promote software success to empirical validation of the Theory of Agencity.

## Canonical core

- [x] The reference pipeline remains NumPy and preserves the accepted equations.
- [x] `S = sqrt(M^2 + O^2)` is tested.
- [x] `Theta = atan2(O, M)` remains the structural orientation convention.
- [x] `U = (M + iO) / S` and `beta = J U` are applied only for `S > 0`; `beta = 0` at `S = 0`.
- [x] `J = ln((e + D) / (e + S))` is tested directly.
- [x] `b = P_c beta` is tested.
- [x] No numerical epsilon is inserted into a valid canonical denominator.
- [x] Existing deterministic scientific-reference tests are part of the required CI suite.

## Public API freeze

- [x] Stable v1.0 entry points are documented in `stable_api.md`.
- [x] Stable API evolution follows Semantic Versioning after 1.0.0.
- [x] Physical parameters are explicit and validation errors are deterministic.
- [x] Compatibility aliases are separated from canonical definitions.
- [x] Experimental/research interfaces are labelled and are not silently promoted.

## Analysis

- [x] Diagnostics consume computed results rather than redefining the core.
- [x] Real-agencity thresholds remain contextual diagnostics.
- [x] `beta != 0` is not used as the definition of real agencity.
- [x] Phase unwrapping, regime classification, event filters, and persistence rules remain outside canonical computation.

## Reproducibility

- [x] Results retain `xi`, `u`, canonical intermediate arrays, `A_ref`, `tau`, `w`, and `P_c`.
- [x] Unit and domain metadata are preserved.
- [x] Backend request/resolution and canonical backend are recorded.
- [x] New computations record the producing AgencityLab version.
- [x] Complex `beta` and `b` survive JSON round-trip and CSV export uses explicit real/imaginary columns.
- [x] Documentation does not promise unique reconstruction of the input observable from `b`.

## User workflows

- [x] Quickstart uses only declared dependencies.
- [x] Minimal scalar example is executable.
- [x] Full scientific workflow example is exercised in CI with its optional dependencies.
- [x] Multiscale public example is executable.
- [x] End-to-end tests cover input -> compute -> diagnostics -> export -> restore.
- [x] Batch, full-history streaming, and multiscale/scalar equivalence are release gates.
- [x] The built-distribution gate exercises only public APIs for install -> import -> compute -> diagnostics -> export.

## Packaging and compatibility

- [x] Package/runtime/documentation release version is `1.0.0`.
- [x] Python 3.10, 3.11, and 3.12 are the supported matrix.
- [x] Minimal runtime depends only on NumPy.
- [x] Optional extras are isolated.
- [x] Wheel and source distribution are built and clean-installed by CI.
- [x] `pip check` runs on clean installations.
- [x] Documentation is built by CI with warnings treated as errors.
- [x] Package classifier identifies the release as stable rather than Alpha/Beta.
- [x] The repository's declared MIT licence contains the complete MIT licence text.
- [x] `CITATION.cff` records the software version, release date, repository, author metadata already present in the project, and MIT licence without inventing a DOI or affiliation.

## Performance and numerical regression

- [x] The retained v0.8 benchmark remains a numerical-equivalence gate for CRM, the canonical pipeline, and `Sigma_Theta`.
- [x] Full-history streaming and threaded batch equivalence remain checked.
- [x] Timing observations are not converted into fragile universal CI thresholds.

## Experimental / deferred

The following do not block 1.0 provided they remain accurately labelled:

- Numba and JAX as experimental primitive layers rather than complete canonical pipelines.
- Riemannian construction without a production pipeline.
- GPU/distributed/out-of-core execution.
- A constant-memory online streaming recurrence.
- Field, extended thermodynamic, quantum, gravitational, and cosmological extensions not included in the stable scalar API.

## Scientific limitations to publish

Version 1.0 remains research software. The Theory of Agencity is still subject to empirical validation and falsification. Results depend on the observable, sampling/preprocessing choices, and physically/contextually justified `A_ref`, `tau`, `w`, and `P_c`. The omission convention `w=tau` is a software convention, not a universal identity. The inverse problem is non-injective, and the current streaming implementation is not a constant-memory recurrence.

## Decision rule

A v1.0 release is a software-readiness decision only when the complete CI is green, no unresolved requested changes remain, documentation and examples build/run, built distributions install cleanly, and no known blocking divergence from the accepted canonical definitions remains.
