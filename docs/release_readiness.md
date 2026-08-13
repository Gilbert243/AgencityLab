# AgencityLab 1.0 release readiness

This checklist is the software release gate for AgencityLab 1.0. It records
implementation and distribution readiness; it does not promote software success
to empirical validation of the Theory of Agencity.

## Canonical core

- [x] The reference pipeline remains NumPy and preserves the accepted equations.
- [x] `S = sqrt(M^2 + O^2)` is tested.
- [x] `Theta = atan2(O, M)` remains the structural orientation convention.
- [x] `U = (M + iO) / S` and `beta = J U` are applied only for `S > 0`; `beta = 0` at `S = 0`.
- [x] `J = ln((e + D) / (e + S))` is tested directly.
- [x] `b = P_c beta` is tested.
- [x] No numerical epsilon is inserted into a valid canonical denominator.
- [x] Diagnostics and historical compatibility wrappers are not part of `agencitylab.core`.

## First stable API contract

- [x] Stable 1.0 entry points are documented in `stable_api.md`.
- [x] Stable API evolution follows Semantic Versioning starting with 1.0.0.
- [x] Pre-1.0 development snapshots create no compatibility aliases in 1.0.
- [x] Physical parameters are explicit and validation errors are deterministic.
- [x] Experimental/research interfaces are labelled and are not silently promoted.
- [x] Package-root discovery is small and namespace-first.

## Analysis

- [x] Diagnostics consume computed results rather than redefining the core.
- [x] Diagnostic products are separate from `AgencityResult`.
- [x] Real-agencity thresholds remain contextual diagnostics.
- [x] `beta != 0` is not used as the definition of real agencity.
- [x] Phase unwrapping, regime classification, event filters and persistence rules remain outside canonical computation.

## Reproducibility

- [x] Results retain `xi`, `u`, canonical intermediate arrays, `A_ref`, `tau`, `w`, and `P_c`.
- [x] Unit and domain metadata are preserved.
- [x] New computations record the producing AgencityLab version.
- [x] Stable result serialization begins at schema `1.0`.
- [x] Complex `beta` and `b` survive JSON round-trip and CSV export uses explicit real/imaginary columns.
- [x] pandas/xarray adapters are isolated from the canonical result model implementation.
- [x] Documentation does not promise unique reconstruction of the input observable from `b`.

## User workflows

- [x] Quickstart uses only declared dependencies.
- [x] Minimal scalar example is executable.
- [x] Full scientific workflow example is exercised in CI with its optional dependencies.
- [x] Multiscale public example is executable.
- [x] End-to-end tests cover input -> compute -> diagnostics -> export -> restore.
- [x] Batch, full-history streaming, and multiscale/scalar equivalence are release gates.
- [x] The built-distribution gate exercises public APIs after clean installation.

## Quality and compatibility

- [x] Package/runtime/documentation release version is `1.0.0`.
- [x] CPython 3.10, 3.11, 3.12, 3.13 and 3.14 are in the test matrix.
- [x] The declared minimum NumPy core contract is tested separately on Python 3.10.
- [x] Minimal runtime depends only on NumPy.
- [x] The distribution ships `py.typed` and the stable public surface has a mypy gate.
- [x] Ruff performs repository correctness checks plus a stronger stable-surface lint gate.
- [x] Coverage is measured and archived without inventing an arbitrary threshold before establishing a baseline.
- [x] Optional extras are isolated and smoke-tested independently.
- [x] Wheel and source distribution are built, checked and clean-installed by CI.
- [x] `pip check` runs on clean installations.
- [x] Documentation is built by CI with warnings treated as errors.
- [x] Package classifier identifies the software contract as stable.
- [x] `CITATION.cff` records version, release date, repository, author metadata and MIT licence.

## Release engineering

- [x] `RELEASING.md` documents the release procedure.
- [x] `SUPPORT.md` documents Python support and the post-1.0 API lifecycle.
- [x] GitHub Release publishing requires an exact `v<package-version>` tag match.
- [x] Wheel and sdist pass `twine check` before publication.
- [x] PyPI publication uses GitHub OIDC Trusted Publishing rather than a long-lived API token.
- [x] The publish job requests `id-token: write` only where required and is scoped to the `pypi` environment.
- [ ] The external PyPI Trusted Publisher / GitHub `pypi` environment must be configured by repository maintainers before the first actual publication.

## Performance and numerical regression

- [x] The retained pre-1.0 benchmark remains a numerical-equivalence gate for CRM, the canonical pipeline, and `Sigma_Theta`.
- [x] Full-history streaming and threaded batch equivalence remain checked.
- [x] Timing observations are not converted into fragile universal CI thresholds.

## Experimental / research boundaries

The following do not block 1.0 provided they remain accurately labelled:

- Numba and JAX as experimental primitive layers rather than complete canonical pipelines.
- Riemannian construction without a production pipeline.
- GPU/distributed/out-of-core execution.
- A constant-memory online streaming recurrence.
- Field, extended thermodynamic, quantum, gravitational, and cosmological extensions outside the stable canonical scalar contract.

## Scientific limitations

Version 1.0 is a software-stability milestone. Scientific conclusions remain
conditional on the observable, sampling/preprocessing choices and
physically/contextually justified `A_ref`, `tau`, `w`, and `P_c`. The omission
convention `w=tau` is a software convention, not a universal identity. The
inverse problem is non-injective, and the current streaming implementation is
not a constant-memory recurrence.

## Decision rule

A 1.0 release is ready only when the complete CI is green, no unresolved
requested changes remain, documentation and examples build/run, built
distributions install cleanly, and no known blocking divergence from the
accepted canonical definitions remains.
