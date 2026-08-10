# Changelog

All notable changes to AgencityLab are documented here.

## 0.2.0 - 2026-08-10

### Canonical Core

- Reconciled the scalar reference pipeline with the current second-edition theory: `u -> u* -> X* -> A* -> M,O -> D,S -> J,U -> beta -> b`.
- Implemented exact canonical normalization `u* = u / A_ref` and reduced time `t* = t / tau`.
- Corrected memory to `M = CRM[u*]` and organisation to the cross-correlation `O = CRM[u*, X*]`.
- Fixed the canonical CRM window to `w = tau`; removed historical `tau / A_fact` compression from the canonical path.
- Removed historical `tanh` saturation from canonical memory and organisation.
- Implemented `D = sqrt(X*^2 + (A* X*)^2)` and `S = sqrt(M^2 + O^2)` without clipping or saturation.
- Implemented `J = ln((e + D)/(e + S))` with the theoretical constant `e = exp(1)` and no machine epsilon inserted into the equation.
- Implemented `U = (M + iO)/S` for `S > 0`, with the explicit canonical branch `U = beta = 0` for `S = 0`.
- Kept `b = P_c beta` exact and linear in characteristic power.
- Made `A_ref`, `tau`, and `P_c` explicit physical/contextual parameters: canonical resolution uses explicit values, metadata, documented physical energetics, or deliberately registered conventions, never silent signal-statistical fallbacks.
- Added an exact null/rest-state precheck. An exactly constant sampled observable bypasses derivative and CRM evaluation and returns `X*=A*=M=O=D=S=J=U=beta=b=0` exactly, following the project maintainer's canonical postulate interpretation rather than trying to prove rest numerically through finite differences.
- Kept numerical epsilon and generic safeguard helpers outside valid canonical equations.

### Tests and traceability

- Added analytical tests for normalization, reduced derivatives, CRM auto/cross behaviour, zero-variance CRM, tiny non-zero signals, canonical `M/O`, dynamic/structural norms, logarithmic contrast, orientation, `S = 0`, `beta`, and `b`.
- Added tiny-positive-value tests to ensure machine epsilon does not redefine physical zero.
- Added tests proving that the exact rest state bypasses derivative and CRM operators.
- Updated smoke, integration, regression, README, overview, and theory mapping for explicit canonical physical parameters and the v0.2 pipeline.

### Scientific boundary

- Version 0.2.0 is an implementation-fidelity milestone, not empirical validation of the Theory of Agencity.
- Coherence, regimes, real-agencity diagnostics, multiscale optimisation, and speculative field/quantum/cosmological extensions remain separate from the canonical scalar core.

## 0.1.12 - 2026-08-10

### Foundations

- Synchronized package metadata and runtime `__version__` at `0.1.12`.
- Corrected the declared Python requirement to `>=3.10` after CI exposed the existing use of `dataclass(slots=True)`, which is not supported by Python 3.9.
- Corrected pytest discovery to use the repository-level `tests/` suite.
- Established a correctness-focused Ruff baseline across the package and tests and modernized the Ruff configuration.
- Added build tooling to the development dependencies.
- Added an `export` optional dependency group for Excel and PDF export backends.
- Replaced the self-referential `all` extra with explicit optional dependencies.
- Added GitHub Actions CI for Python 3.10 and 3.11 with import, Ruff, pytest, and package-build checks.
- Added foundation tests for package metadata and the public compute API.
- Repaired the characteristic-time API wiring so the compute pipeline can pass structural metadata context without failing at runtime.
- Updated smoke, integration, regression, and analysis tests to exercise the public API with explicit scientific input keywords.
- Reworked the README, contribution guide, overview, and theory mapping to state the alpha/research status clearly and remove unresolved merge-conflict markers.
- Documented known differences between the then-current theory sources and legacy numerical choices in memory, organisation, and CRM handling.

### Deferred maintenance debt

- Broader Ruff style cleanup remains outside the v0.1 scientific-foundation change so formatting and naming churn do not obscure theory reconciliation.

### Superseded by 0.2.0

The scientific reconciliation items deferred from 0.1.12 were resolved in 0.2.0 according to the current second-edition theory. Historical Git formulations are not used to define canonical physics.

## 0.1.0

- Initial project skeleton.
