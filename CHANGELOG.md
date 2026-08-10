# Changelog

All notable changes to AgencityLab are documented here.

## 0.6.0 - 2026-08-10

### Multiscale & Extensions

- Replaced the legacy multiscale implementation with an explicit extension engine that preserves the canonical scalar equations and never silently compresses the CRM window or infers physical parameters from signal statistics.
- Added a time-resolved `b(t, tau)` spectrum with explicit `tau` and `w` coordinates. The default extension convention is `w = tau` at every scale; independent `w` is available only through an explicitly named advanced-theory path.
- Kept stable `compute_agencity()` unchanged: its project-canonical scalar contract still requires `w = tau` exactly.
- Added the advanced-theory Chapter 13 angular-stability window optimiser `Phi2(w)`, with discrete candidate windows `w = N delta` and explicit exclusion of candidates whose structural orientation is undefined because no complete interval has `S > 0`.
- Added `compute_discrete_agencity()` as a sampled-sequence convenience API that constructs a uniform coordinate and delegates to the existing canonical discrete implementation instead of duplicating equations.
- Added the advanced multivariate construction: scalar Agencity is computed independently per component, `beta_multi` is the pointwise `P_c`-weighted average, and `b_total` is the vector-additive sum of component fluxes. Scalar, per-component, and sampled component-power inputs are supported.
- Reworked analysis-side multiscale helpers to require explicit `A_ref` and `P_c`, to reject physical normalization of `b`, and to label `find_optimal_tau()` as diagnostic selection across a supplied scale grid rather than inference of the physical characteristic time.
- Added `docs/multiscale_extensions.md` and dedicated extension tests covering spectrum/scalar equivalence, independent-window isolation, window optimisation, sampled-signal equivalence, `P_c`-weighted multivariate aggregation, time-varying component power, and the Riemannian implementation boundary.
- Added public APIs `compute_agencity_spectrum()`, `optimize_agencity_window()`, `compute_discrete_agencity()`, `compute_multivariate_agencity()`, and `riemannian_extension_status()`.
- Bumped the package version to `0.6.0`; the canonical `AgencityResult` serialization schema remains unchanged.

### Scientific boundary

- `tau`, CRM window `w`, sampling interval `delta`, and a multiscale scan are separate concepts. A spectrum maximum is not automatically the physical `tau`, and an optimised signal-derived `w` is not silently promoted to a canonical structural parameter.
- The multivariate extension follows the accepted `P_c`-weighted formula and additive flux law; it does not invent a coupled-vector CRM theory.
- The advanced source sketches a Riemannian formulation but explicitly defers detailed analysis. v0.6 therefore does not invent a production Riemannian pipeline and reports that extension as `experimental` and unimplemented.
- No v0.6 extension changes canonical `beta`, `J`, CRM, `M`, `O`, `D`, `S`, `P_c`, `A_ref`, or the exact rest-state convention.

## 0.5.0 - 2026-08-10

### Agencity Analysis

- Established a strict analysis layer that consumes canonical `AgencityResult` outputs without recomputing or redefining `A_ref`, `tau`, CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b`.
- Implemented the theory-derived local angular variance `Sigma_Theta(t) = Var(Theta(s); s in [t-tau,t])` on complete structurally valid windows. Circular resultant variance remains a separately labelled diagnostic rather than a replacement for `Sigma_Theta`.
- Reworked structural coherence to use canonical `Theta = atan2(O, M)` and to exclude `S = 0` points where orientation is physically undefined. `arg(beta)`/`arg(b)` is not substituted for structural orientation because the sign of `J` can add a pi phase shift.
- Reworked the real-agencity criterion around `S > 0`, low `Sigma_Theta`, and significant `|b|` without universal thresholds. Local evaluation requires explicit contextual thresholds; a global Boolean additionally requires an explicit persistence fraction.
- Replaced the legacy epsilon-modified trajectory curvature with the signed algebraic curvature of the intrinsic `beta(t)` curve, leaving curvature undefined when `beta_dot = 0` instead of inserting epsilon into the denominator.
- Added structural winding diagnostics from canonical `Theta`, including explicit undefined handling across structural zeros and raw finite-interval winding without forced integer quantisation.
- Added exact agencity-zero detection from `S = 0` or `J = 0`, critical-surface crossings `D = S`, and explicit-threshold wrapped `Theta` jump detection.
- Added local peak diagnostics for canonical dynamic intensity `D` and explicit-threshold structural plateaus of `S`.
- Added threshold-free regime signatures and contextual `RegimeCriteria` classification for the theory table (`null`, `passive_damped`, `active_oscillating`, `unstable`, `stochastic`, `chaotic`). Non-null classification defaults to `undetermined` when criteria are not supplied.
- Made multiscale scaling signatures reject non-positive log inputs instead of replacing them by epsilon. Qualitative slope interpretation now requires an explicit diagnostic threshold.
- Added `analyze_coherence()`, `analyze_geometry()`, and `analyze_regime_signature()` to the public analysis API while preserving existing analysis entry points.
- Added `docs/agencity_analysis.md` and dedicated analysis tests for `Sigma_Theta`, real-agencity threshold policy, beta curvature, winding, exact zeros, transitions, D peaks, regime criteria, signatures, and non-mutation of canonical arrays.
- Bumped the package version to `0.5.0`; the `AgencityResult` serialization schema remains `0.3` because the canonical result model is unchanged.

### Scientific boundary

- Version 0.5.0 creates scientific diagnostics from canonical outputs; it does not modify the v0.2 equations and is not empirical confirmation of the theory.
- Numerical criteria for "low" angular variance, "significant" flow, peak prominence, structural plateaus, persistence fractions, and regime classification are contextual diagnostics, not universal constants.
- Noise and chaos may contain local non-zero `beta` and may satisfy a local real-agencity criterion intermittently; the API does not convert such a local event into a whole-record claim without an explicit persistence rule.
- Geometry is computed from intrinsic `beta`, not from `b`, so an externally varying `P_c(t)` cannot silently alter the state-trajectory curvature or winding analysis.

## 0.4.0 - 2026-08-10

### Scientific validation

- Added a deterministic scientific reference battery covering exact rest, a sinusoid, an underdamped passive oscillator, the Van der Pol oscillator, a negative-damping unstable oscillator, a low-pass-filtered Ornstein-Uhlenbeck process, and the classical Lorenz system.
- Added theory-facing regime checks for exact nullity, periodic structure, passive structure-dominated tails, bounded self-sustained oscillation, unstable logarithmic-contrast growth, and irregular stochastic/chaotic orientation.
- Added mathematical-property tests for state-translation invariance, global sign-inversion invariance, temporal covariance under simultaneous time/tau rescaling, small structured-amplitude behaviour, large-amplitude logarithmic growth, and exact linearity in characteristic power.
- Added a uniform-refinement convergence experiment against a fine-grid reference and a smooth-perturbation robustness experiment. These are numerical validation checks, not replacements for the analytical theorems.
- Added explicit finite-record CRM warm-up handling to validation metrics (`t >= t0 + 2*tau`) without changing canonical CRM or introducing prehistory into the core.
- Added `docs/scientific_validation.md`, distinguishing theorem-level claims, fixed-benchmark numerical observations, numerical tolerances, and empirical validation.
- Bumped the package version to `0.4.0` without changing the v0.2 canonical equations or the v0.3 stable API schema.

### Scientific boundary

- Version 0.4.0 validates the implementation against selected consequences and reference regimes stated by the accepted theory; it is not experimental confirmation that Agencity is a universal physical observable.
- Benchmark tolerances are fixed numerical acceptance criteria, never universal thresholds for coherent or "real" agencity.
- Filtered noise is explicitly allowed to have non-zero `D` and local non-zero `beta`; stochastic validation focuses on reproducible structural/orientational behaviour rather than forcing a null signal.
- Current canonical `e = exp(1)` governs v0.4 tests. Earlier numerical examples using a different effective offset convention are not silently converted into current acceptance targets.

## 0.3.0 - 2026-08-10

### Stable computational API

- Stabilized `compute_agencity()` as the scalar-signal canonical reference entry point without changing the v0.2 equations.
- Added strict one-dimensional input validation, explicit ambiguity errors for `u` versus the compatibility alias `data`, and rejection of unknown compute keywords instead of silently ignoring them.
- Added typed public exceptions for validation, physical-parameter, unit-label, batch, and streaming failures while keeping validation exceptions compatible with existing `ValueError` handling.
- Added descriptive unit-label support: `unit` for `u`/`A_ref`, `coordinate_unit` for `xi`/`tau`, and `power_unit` for `P_c`; observable flux `b` is labelled as informational power (`power_unit·nat`, e.g. `W·nat`). No hidden unit conversion is performed.
- Stabilized `ExperimentMetadata` with validation, unit contracts, canonical `memory_window`, forward-compatible unknown-field preservation, and explicit separation of legacy observational metadata from canonical modifiers.
- Stabilized `AgencityResult` with schema version `0.3`, scalar or sampled strictly positive `P_c`, consistent metadata synchronization, complex round-tripping, canonical wrapped `theta = angle(U)`, and exact `eta = |b| / P_c` without epsilon substitution.
- Restored explicit support for externally supplied time-varying `P_c(t)` as a sampled profile or callable, preserving the canonical `b(t) = P_c(t) beta(t)` relation without deriving power from the observed signal.
- Preserved compatibility fields and aliases: `data=`, historical `Pc=`, legacy serialized physical-field names, legacy `metadata.extra["memory_window"]`, and summary keys including `Pc_mean`, `A_fact`, and `resolution_scale`.
- Improved deserialization so legacy payloads may recover `A_ref`, `tau`, and scalar `P_c` from metadata before any compatibility default is considered.
- Improved batch execution with per-item physical parameters, metadata/config overrides, deterministic ordering, and indexed `BatchItemError` failures.
- Improved streaming with persistent physical context, continuous implicit coordinates across chunks, explicit coordinate-order validation, and `StreamNotReadyError` when two CRM windows are not yet available.
- Repaired fluent-pipeline compatibility so `set_tau()` and `set_power()` affect the physical metadata actually used by computation. `set_resolution_scale()` remains observational metadata only, and `set_activity_factor()` is deprecated metadata that does not modify canonical CRM.
- Added a dedicated v0.3 stable-API test suite and user documentation.

### Scientific boundary

- Version 0.3.0 is an API-stability milestone, not a change to canonical physics and not empirical validation of the Theory of Agencity.
- `beta`, `J`, CRM, `M`, `O`, `D`, `S`, `tau`, `w`, `P_c`, and `A_ref` are not redefined for software convenience.
- Time-varying `P_c(t)` support is an implementation of the canonical multiplicative flux relation, not a signal-derived power estimator.

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
