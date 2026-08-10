# v1.0 release readiness — v0.9 Release Candidate

This checklist is the software release gate for AgencityLab 1.0. It does not promote implementation success to empirical validation of the Theory of Agencity.

## Canonical core

- [x] The reference pipeline remains NumPy and preserves the accepted equations.
- [x] `S = sqrt(M^2 + O^2)` is tested.
- [x] `U = (M + iO) / S` and `beta = J U` are applied only for `S > 0`; `beta = 0` at `S = 0`.
- [x] `b = P_c beta` is tested.
- [x] No numerical epsilon is inserted into a valid canonical denominator.
- [x] Existing deterministic scientific-reference tests are part of the required CI suite.

## Public API

- [x] Stable candidate-v1.0 entry points are documented in `stable_api.md`.
- [x] Physical parameters are explicit and validation errors are deterministic.
- [x] Compatibility aliases are separated from canonical definitions.
- [x] Experimental interfaces are labelled and are not silently promoted.

## Analysis

- [x] Diagnostics consume computed results rather than redefining the core.
- [x] Real-agencity thresholds remain contextual diagnostics.
- [x] Phase unwrapping, regime classification, event filters, and persistence rules remain outside canonical computation.

## Reproducibility

- [x] Results retain `xi`, `u`, canonical intermediate arrays, `A_ref`, `tau`, `w`, and `P_c`.
- [x] Unit and domain metadata are preserved.
- [x] Backend request/resolution and canonical backend are recorded.
- [x] New computations record the producing AgencityLab version.
- [x] Complex `beta` and `b` survive JSON round-trip and CSV export uses explicit real/imaginary columns.

## User workflows

- [x] Quickstart uses only declared dependencies.
- [x] Minimal scalar example is executable.
- [x] Full scientific workflow example is exercised in CI with its optional dependencies.
- [x] Multiscale public example is executable.
- [x] End-to-end tests cover input -> compute -> diagnostics -> export -> restore.
- [x] Batch, full-history streaming, and multiscale/scalar equivalence are release gates.

## Packaging and compatibility

- [x] Python 3.10, 3.11, and 3.12 are the supported matrix.
- [x] Minimal runtime depends only on NumPy.
- [x] Optional extras are isolated.
- [x] Wheel and source distribution are built and clean-installed by CI.
- [x] `pip check` runs on clean installations.
- [x] Documentation is built by CI with warnings treated as errors.

## Experimental / deferred

The following do not block 1.0 provided they remain accurately labelled:

- Numba and JAX as experimental primitive layers rather than complete canonical pipelines.
- Riemannian construction without a production pipeline.
- GPU/distributed/out-of-core execution.
- A constant-memory online streaming recurrence.
- Field, thermodynamic, quantum, gravitational, and cosmological extensions not included in the stable scalar API.

## Decision rule

A v1.0 release is a software-readiness decision only when the complete Release Candidate CI is green, no unresolved requested changes remain, documentation and examples build/run, and no known blocking divergence from the accepted canonical definitions remains.
