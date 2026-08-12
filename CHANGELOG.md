# Changelog

All notable changes to AgencityLab are documented here.

## 1.1.4 - 2026-08-12

### Added

- Added the explicitly speculative Chapter-21 quantum Agencity layer for the autonomous `phi` field: broken-symmetry radial and Goldstone masses and dispersion relations, caller-regulated propagators, finite Fock-space primitives, occupation expectations, the constant-bridge Agencity uncertainty bound, and the explicitly stated leading one-loop quartic beta-function term.
- Added the explicitly speculative homogeneous flat-FLRW cosmology application from Volume 2 Chapters 20 and 22: shared field energy density and pressure, equation-of-state evaluation, Friedmann and acceleration-equation residuals, explicit expanding/contracting initial branches, and a deterministic RK4 reference solver.
- Added selected top-level quantum and cosmology APIs while retaining the complete contracts under `agencitylab.quantum` and `agencitylab.applications.cosmology`.
- Added public quantum/cosmology documentation and cross-layer release tests connecting both extensions to the shared `QuarticAgencityPotential` without duplicating field physics.

### Scientific status

- The scalar `u -> beta -> b` pipeline remains canonical and unchanged.
- Observable spatial orchestration and generic field numerics remain `experimental`.
- Autonomous `phi`, classical field dynamics, coherent structures, thermodynamics, and classical gravity remain `research`.
- Quantum/agenton primitives and homogeneous cosmology are `speculative`; implementation and numerical tests are not experimental evidence for quantum Agencity, agentons, inflation, or dark energy.
- The quantum layer quantises only the proposed autonomous `phi` field and does not quantise or redefine the canonical observable pipeline.

### Numerical and theoretical boundaries

- Finite Fock matrices are explicit numerical truncations. The unavoidable highest-state commutator defect is exposed rather than hidden or presented as the exact infinite-dimensional bosonic algebra.
- Propagator `epsilon` is caller supplied; no machine epsilon is inserted as a physical regulator.
- The Agencity uncertainty helper is restricted to the constant-parameter bridge where the source relation is operationally unambiguous; no derivative of time-varying bridge factors is silently omitted.
- The FLRW solver initializes `H` from the first Friedmann equation and then reports its constraint residual during RK4 evolution instead of projecting the numerical state back onto the constraint.
- The minimal broken-symmetry quartic vacuum remains negative, `V_min = -lambda^2/(4 mu)`. AgencityLab does not add a positive vacuum offset, cosmological constant, or modified potential merely to reproduce positive dark-energy density.
- No path-integral engine, vacuum-energy prescription, scattering engine, lattice QFT, generic quantum gravity, cosmological perturbation/CMB machinery, inflationary fit, or observational dark-energy inference is introduced.
- NumPy remains the only required runtime dependency and the stable canonical scalar contract remains protected.

## 1.1.3 - 2026-08-12

### Added

- Added the unified Agencity thermodynamics research layer: field dissipation and entropy-production evaluators, temperature-dependent `lambda(T)`, distinct field and contrast agencial entropies, energy-balance and second-law residuals, Modulus Law and Phase Law evaluators, and the explicitly conditional Volume-1 Landauer relations.
- Added the named empirical Phase-Law reference fit (`alpha ~= 0.82`, `beta_fit ~= -1.50`, `R^2 ~= 0.87`) without making those values universal defaults.
- Added the classical gravity research layer with explicit Chapter-19 `(-,+,+,+)` geometry, matter/action-density evaluators, external U(1) gauge covariant derivative, minimal stress-energy, Einstein-equation residual, nonminimal action/field coupling, and named minimal/conformal coupling helpers.
- Added public documentation for thermodynamics and gravity and selected top-level research exports while retaining the complete APIs under `agencitylab.thermodynamics` and `agencitylab.gravity`.
- Added cross-layer release tests connecting dissipative field solutions to thermodynamic dissipation, checking vacuum zero-dissipation behaviour, verifying shared use of `QuarticAgencityPotential`, protecting the gravity metric-signature contract, and checking U(1) invariance of field entropy.

### Scientific status

- The scalar `u -> beta -> b` pipeline remains canonical and unchanged.
- Observable spatial orchestration and generic field numerics remain `experimental`.
- Autonomous `phi`, classical field dynamics, coherent structures, thermodynamics, and classical gravity remain `research` and are not presented as empirically validated physics.
- Quantum/agenton and cosmological extensions remain outside this release and retain `speculative` status.
- The Chapter-16 `(+,-,-,-)` flat-field convention and Chapter-19 `(-,+,+,+)` gravity convention remain explicitly distinct. AgencityLab does not silently rewrite either source equation to force equality.

### Limitations and compatibility

- The gravity package does not implement a generic Einstein/3+1 solver, cosmological evolution, autonomous gauge dynamics, or the complete nonminimal stress-energy tensor because the accepted source does not fully specify that tensor.
- Thermodynamic balance laws are evaluated rather than imposed; no autonomous thermal solver, invented `J_E` discretisation, universal Phase-Law fit, or closed autonomous equation for `b` is introduced.
- The two source-layer agencial entropies remain separate APIs and the historical Shannon-style `agential_entropy` helper remains only a deprecated legacy placeholder.
- NumPy remains the only required runtime dependency and the stable canonical scalar contract is unchanged.

## 1.1.2 - 2026-08-12

### Added

- Added the classical autonomous-field research dynamics from Volume 2: conservative Klein-Gordon, dissipative Klein-Gordon, and overdamped TDGL, all reusing the shared quartic potential and NumPy field operators introduced in 1.1.1.
- Added deterministic fixed-step simulators returning `DynamicalAgencityFieldSolution`: velocity-Verlet for conservative KG, RK4 on the first-order system for dissipative KG, and RK4 for TDGL.
- Added research coherent-structure references: the rescaled real-sector/Z2 domain wall, two-dimensional U(1) vortex construction with caller-supplied radial profile, radial residual evaluation, spatial phase winding, and explicit zero/near-zero masks.
- Added cross-layer integration tests proving the static domain-wall residual is the same `laplacian(phi) - potential.gradient(phi)` used by the KG acceleration and that the broken vacuum remains stationary under all three reference dynamics.
- Exposed the dynamics and coherent-structure APIs through `agencitylab.fields` and selected top-level `agencitylab` imports.
- Added `docs/classical_field_dynamics.md` and `docs/coherent_structures.md` and updated the field-foundation documentation.

### Scientific status

- The scalar `u -> beta -> b` pipeline remains canonical and unchanged.
- Observable spatial orchestration remains `experimental`.
- Autonomous `phi`, the quartic field dynamics, domain-wall/vortex references, and spatial field topology remain `research`; numerical agreement is implementation evidence, not empirical validation.
- The real kink is explicitly a real-sector/Z2 reference and is not claimed to be a generally stable topological wall of the full complex U(1) theory.
- The vortex radial profile is not invented: callers provide `f(r)`, while AgencityLab evaluates the source ansatz/residual and winding diagnostics.

### Compatibility and cleanup

- Retired the old ambiguous `field_rhs`, `solve_field`, `zero_boundary`, `field_energy`, and empty `detect_domain_walls` placeholders instead of silently mapping them to one research equation or heuristic.
- Kept the unsupported generic discretized `action()` boundary explicit; 1.1.2 implements the potential and equations of motion but does not fabricate a variational discretization.
- Documented that experimental/research interfaces may evolve within the 1.1.x development line while the stable canonical scalar contract remains protected.
- NumPy remains the only required runtime dependency.

## 1.1.1 - 2026-08-12

### Added

- Integrated the shared dynamical-field foundations: explicit research bridge `phi = sqrt(P_c * tau) * beta`, quartic potential, broken-vacuum primitives, field-energy primitives, scientific-status/provenance models, uniform N-D grids, spatial operators, boundary contracts, generic RK4/velocity-Verlet integrators, and CFL diagnostics.
- Added `DynamicalAgencityFieldState` and `DynamicalAgencityFieldSolution` without conflating them with `ObservableAgencityFieldResult`.
- Added the common `ScientificStatus` taxonomy (`canonical`, `experimental`, `research`, `speculative`) and parameter-provenance contracts.
- Added public documentation for the explicit observable-to-autonomous-field boundary and the Volume-2 source-term convention relative to standard Wirtinger normalization.

### Scientific status

- The `beta_obs -> phi` bridge is explicit and `research`; `compute_agencity_field()` never performs it automatically.
- Generic field numerics are `experimental` infrastructure and do not define Agencity physics.
- Autonomous field physics is `research`; no PDE time evolution, thermodynamics, gravity, quantization, or cosmology was claimed in 1.1.1.
- `P_c = 0` gives `phi = 0` exactly with no epsilon substitution or implicit inverse.

## 1.1.0 - 2026-08-11

### Added

- Added experimental observable spatial Agencity fields through `compute_agencity_field()`, applying the reference scalar `compute_agencity()` pipeline independently at every spatial location.
- Added `ObservableAgencityFieldResult` with original field geometry, spatial coordinates, all local canonical intermediate quantities, resolved local physical parameters, scientific status, backend, and reproducibility metadata.
- Added scalar and spatial `A_ref(x)`, `tau(x)`, and `w(x)`, plus scalar, spatial, and spatio-temporal `P_c(x,t)`, with explicit shape resolution and no accidental NumPy broadcasting.
- Added pointwise scalar-equivalence, multidimensional geometry, `time_axis`, local rest, local `P_c=0`, invariance, validation, and NumPy-only packaging tests.
- Added `docs/observable_fields.md` documenting the observable-field contract and the strict separation from the future dynamical field.

### Scientific status

- `compute_agencity_field()` is **experimental spatial orchestration over the canonical scalar pipeline**. It is not new canonical field physics.
- CRM remains temporal only and is evaluated independently at each spatial location. v1.1 introduces no spatial CRM, smoothing, derivative, Laplacian, autonomous `phi`, PDE, potential, coherent-structure dynamics, thermodynamics, gravity, quantum field, or cosmology.
- The canonical branch `S=0 => U=0 => beta=0` and exact `b=P_c*beta` relation are inherited unchanged from `compute_agencity()`; no epsilon-based physics was introduced.
- `tau` and `w` remain distinct. When `w` is omitted, `w=tau` is used only as the explicit software fallback and is recorded as such.

### Compatibility and placeholders

- Retained historical `AgencityField` only as a compatibility alias for `ObservableAgencityFieldResult`, avoiding ambiguity with the future dynamical field type.
- Replaced misleading historical PDE, solver, boundary-condition, field-energy, action, and domain-wall placeholders with explicit `NotImplementedError` boundaries reserved for the v1.2 research milestone.
- NumPy remains the reference and only required dependency for the observable-field API.

## 1.0.1 - 2026-08-11

### Fixed

- Corrected the stable characteristic-power domain from strictly positive to finite non-negative: `P_c >= 0`. Exact zero is now accepted for scalar, sampled, and multivariate power inputs and preserves the canonical identity `P_c = 0 => b = 0` without epsilon substitution.
- Removed the historical `compute_full_agencity()` duplicate physical orchestration. It remains as a deprecated compatibility wrapper delegating to the sole reference canonical pipeline, `compute_agencity()`, and no longer rejects explicit `w != tau`.
- Reclassified historical coherence and real-agencity helpers under `agencitylab.core` as deprecated legacy diagnostics; `agencitylab.analysis` remains the reference interpretation layer.
- Reworked `compute_discrete_agencity()` to implement the explicit Volume-2 centered first and direct centered second differences rather than silently using the successive `gradient -> gradient` approximation of the sampled continuous pipeline.

### Scientific conformance

- Locked the maintainer decision that `tau` and CRM width `w` are distinct; omission of `w` uses only the explicit software fallback `w=tau`, now recorded in result metadata.
- Confirmed `A_ref` as a fixed physical/contextual reference amplitude with no signal-statistical fallback in the canonical path.
- Confirmed `e = exp(1)` in `J = ln((e+D)/(e+S))`; historical `(1+D)/(1+S)` variants remain legacy divergences only.
- Preserved the exact branch `S=0 => U=0 => beta=0` and kept numerical epsilon outside valid physical equations.
- Documented source-layer historical tensions rather than changing theory to fit code.

### Discrete validation

- Added exact constant, linear, and quadratic stencil tests, including the `u(t)=t^2` constant-second-derivative case.
- Added analytical sinus transfer tests showing the Volume-2 direct second difference has amplitude factor `4 sin^2(z/2)/z^2`, distinct from the successive-first-difference factor `(sin(z)/z)^2`.
- Added second-order convergence and boundary checks plus downstream propagation through `D`, `S`, `J`, `U`, `beta`, and `b`.
- Added deterministic discrete stress coverage for sinusoidal, damped, Van der Pol, unstable, and filtered stochastic signals.

### Compatibility

- The stable public API remains source-compatible; corrected scientific semantics are released as a SemVer patch.
- `AgencityResult.eta` remains the inverse ratio `|b|/P_c` where `P_c>0`; at `P_c=0` it is explicitly undefined and represented as `NaN` instead of adding epsilon or reconstructing `beta` indirectly.
- NumPy remains the complete stable reference backend; optional acceleration and research extensions are unchanged.

## 1.0.0 - 2026-08-11

### Stable

- Established the first stable public software contract around `compute_agencity()`, `AgencityResult`, `ExperimentMetadata`, documented diagnostics, batch, retained-history streaming, multiscale, discrete, multivariate, export, visualization, scientific-workflow, and fluent-pipeline entry points.
- Froze the accepted scalar canonical equations without adding a new scientific feature or changing CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.
- Adopted Semantic Versioning for the stable public contract: patch releases for compatible fixes, minor releases for backwards-compatible additions, and major releases for intentional breaking changes.

### Added

- Added `CITATION.cff` using only author, repository, version, date, and licence metadata already supported by the project; no DOI or affiliation is invented.
- Added explicit v1.0 release gates for stable package metadata, complete MIT licence text, citation metadata, and producer-version consistency.
- Added a built-wheel public end-to-end CI workflow covering clean installation, import, canonical compute, diagnostics, JSON/CSV export, complex result restoration, and `pip check`.

### Changed

- Bumped package/runtime/documentation status from `0.9.0` Release Candidate to `1.0.0` Stable Scientific Release.
- Updated the package classifier from Beta to `Development Status :: 5 - Production/Stable`.
- Finalized the stable API, SemVer, parameter, reproducibility, diagnostic, streaming, multiscale, and research-boundary documentation.
- Updated contribution guidance to the tested Python 3.10/3.11/3.12 matrix and the v1.0 stable-API evolution policy.

### Fixed

- Replaced the incomplete placeholder `LICENSE` body with the complete MIT License text already declared by package metadata and documentation.
- Removed stale Release Candidate/current-version wording from the primary README and Sphinx documentation status pages.

### Deprecated

- No stable v1.0 API is newly removed. Existing legacy compatibility spellings and metadata remain isolated as documented; new code should use the stable canonical spellings and entry points.

### Performance

- No new performance algorithm is introduced in v1.0. The retained v0.8 benchmark remains a numerical-equivalence and regression-observation gate without fragile universal timing thresholds.

### Documentation

- Published the final stable/diagnostic/experimental/research distinction, scientific limitations, SemVer contract, citation guidance, and v1.0 readiness criteria.
- Kept strict Sphinx warnings-as-errors and executable user examples as release gates.

### Compatibility

- Python 3.10, 3.11, and 3.12 remain officially tested.
- Minimal installation remains NumPy-only; SciPy, pandas, Matplotlib, Numba, and JAX remain outside the minimal import path.
- Supported optional extras remain independently installed/smoke-tested, while the `docs` extra is exercised by the strict documentation job.
- Wheel and source-distribution clean installs remain required.

### Scientific validation

- The deterministic reference suite continues to cover exact rest, sinusoidal structure, passive damping, Van der Pol oscillation, negative-damping instability, filtered Ornstein-Uhlenbeck dynamics, and Lorenz dynamics.
- Canonical identities, invariances/limits, CRM numerical equivalence, edge cases, batch/thread equivalence, full-history streaming/one-shot equivalence, and multiscale/scalar equivalence remain release gates.
- Version 1.0 is a stable software release, not empirical confirmation of the Theory of Agencity. Diagnostic thresholds remain contextual, accelerated backends remain experimental, and fundamental extensions remain research/speculative.

### 0.x path to stability

- 0.2 reconciled the canonical scalar engine with the accepted theory; 0.3 stabilized computational results and validation; 0.4 added deterministic scientific validation; 0.5 separated diagnostics from canonical computation; 0.6–0.7 added and reconciled multiscale/extensions and scientific UX; 0.8 hardened engineering/performance and dependency boundaries; 0.9 froze the candidate v1.0 contract and release gates.

## 0.9.0 - 2026-08-10

### Added

- Added explicit candidate-v1.0 public API boundaries for stable, experimental, and legacy/compatibility interfaces.
- Added `ExperimentMetadata.agencitylab_version`; every new `compute_agencity()` result records the producing software version for reproducibility.
- Added release-candidate end-to-end tests covering input, canonical compute, diagnostics, JSON/CSV export, complex result restoration, batch ordering/per-item physics, full-history streaming equivalence, multiscale/scalar equivalence, and critical invalid-input/physical-parameter cases.
- Added a v1.0 release-readiness checklist.
- Added CI gates for Sphinx documentation and critical executable user examples.

### Changed

- Bumped the package and runtime version to `0.9.0` and moved the package classifier from Alpha to Beta for the Release Candidate.
- Reframed the README and documentation around stabilization, reproducibility, and the candidate-v1.0 API contract rather than feature expansion.
- Extended the metadata unit contract so `w` is explicitly labelled with the coordinate unit, alongside `xi` and `tau`.
- Kept NumPy as the stable complete canonical backend; Numba and JAX remain experimental primitive layers.

### Fixed

- Fixed the Sphinx repository root and stale documentation version (`0.1.0`) so docs are built against the actual package and release version.
- Fixed Sphinx 9 intersphinx inventory configuration.
- Corrected stale stable-API documentation that incorrectly claimed an explicit `w != tau` was invalid; Volume 2 and the current implementation keep `w` distinct from `tau`, with `w=tau` only as the omission convention.
- Fixed the quickstart dependency contract: the complete visualization/export walkthrough now declares the `viz` and `export` extras it actually uses.
- Replaced the broken legacy multiscale example with an executable example using `compute_agencity()` and `compute_agencity_spectrum()` with explicit physical parameters.

### Deprecated

- No new stable API is removed in v0.9. Existing legacy compatibility paths such as `activity_factor`/`A_fact`, `resolution_scale`, historical `Pc=`, `data=`, and pipeline-builder aliases remain isolated from canonical physics; new code should use the canonical spellings and stable entry points documented in `docs/stable_api.md`.

### Performance

- No new performance formula or algorithm is introduced. The v0.8 before/after benchmark remains in CI as a scientific-equivalence regression gate; timing is still observational rather than a pass/fail threshold.

### Documentation

- Added warnings-as-errors Sphinx CI and release-readiness documentation.
- Updated README, quickstart, stable API, documentation index, and examples for the 0.9 Release Candidate.

### Compatibility

- Python 3.10, 3.11, and 3.12 remain officially tested.
- Minimal installation remains NumPy-only; supported extras remain isolated and independently smoke-tested.
- Wheel and source-distribution clean-install checks remain required, including `pip check` and a minimal canonical computation.
- The `AgencityResult` serialization schema remains `0.3`; the v0.9 metadata addition is backwards-compatible and older payloads without a producer version remain readable.

### Scientific boundary

- Version 0.9 stabilizes software contracts and does not redefine CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.
- Documentation/build/test success is implementation evidence, not empirical confirmation of the Theory of Agencity.

## 0.8.0 - 2026-08-10

### Engineering & Performance

- Replaced the ordinary canonical CRM Python loop, which recomputed adjacent-window means, variances, and covariance in `O(N*w)`, with an `O(N)` rolling-moment implementation. Exactly constant windows are detected without tolerance, very short and numerically delicate windows fall back to the direct centred Pearson definition, and the finite-record warm-up remains unchanged.
- Retained the pre-v0.8 direct CRM inside a reproducible benchmark suite and added before/after measurements for CRM and the complete canonical pipeline, stage profiling, approximate peak-memory observations, and representative analysis, multiscale, batch, and streaming workloads. Runtime observations are archived by CI but are not fragile timing gates.
- Added direct numerical-equivalence tests for auto- and cross-CRM, constant windows, extreme dynamic ranges, and the complete downstream pipeline. Added a 100,000-sample long-signal test and preserved the exact `S`, `U`, `beta`, and `b` identities.
- Added batch tests for per-item `A_ref`, `tau`, `w`, and `P_c`, deterministic input order, and serial/thread equivalence. Added full-history streaming versus one-shot equality and multiscale-row versus independent-canonical-computation equality.
- Audited the optional NumPy, Numba, and JAX primitives. Removed historical epsilon-based variance classification and denominator modification from their CRM helpers. NumPy is explicitly the stable canonical pipeline; Numba and JAX are explicitly experimental primitive layers and cannot silently replace the reference equations.
- Added `backend_capabilities()` and result metadata recording the requested/resolved optional primitive backend separately from `canonical_backend="numpy"`.
- Reduced the minimal runtime dependency set to NumPy. Moved SciPy validation generators, pandas/xarray adapters, visualization, export, Numba, and JAX support into explicit extras. Retained `ml` only as a compatibility alias for the narrower `numba` and `jax` extras.
- Added Python 3.12 to the tested matrix, clean wheel and source-distribution installation tests, minimal-import/minimal-compute checks without optional dependencies, isolated smoke tests for `data`, `viz`, `export`, `numba`, and `jax`, and archived distribution artifacts.
- Added engineering/performance documentation, updated repository URLs to `somafgroup/AgencityLab`, replaced the obsolete scaling script with the maintained v0.8 benchmark entry point, and bumped the package version to `0.8.0`.

### Scientific boundary

- Version 0.8 changes implementation complexity, dependency boundaries, tests, and observability; it does not redefine CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.
- Numerical fallbacks preserve the centred Pearson coefficient and the exact zero-denominator convention. Machine epsilon is used only to decide when a rolling-moment subtraction is numerically unsafe and must be recomputed directly; it is not inserted into a canonical equation or used as a physical zero threshold.
- Performance measurements are implementation observations on recorded environments, not empirical evidence for the Theory of Agencity and not universal performance guarantees.

## 0.7.0 - 2026-08-10

### Scientific UX

- Added the researcher-facing `scientific_workflow()` orchestration for `signal -> result -> diagnostics -> report -> figures -> exports`, while keeping each artifact separately inspectable through `ScientificStudy`.
- Added scientific overview, intrinsic beta geometry/curvature, real-agencity diagnostic, and theoretical multiscale `b(t,tau)` visualizations; retained compatibility views while fixing complex-value plotting, beta-plane geometry, frequency-spectrum handling, and the previously broken heatmap entry point.
- Added sample-wise `export_result_csv()` with explicit real/imaginary/magnitude columns for complex quantities and `export_study_json()` for reproducible result + analysis + text-report bundles.
- Expanded the human-readable scientific report with `tau`, CRM width `w`, `A_ref`, units, `|b|`, `J`, `Sigma_Theta`, beta curvature, winding, regime status, and real-agencity status.
- Added complete researcher documentation: `docs/scientific_ux.md`, reproducible quickstart and full-pipeline tutorials, a visualization gallery, and deterministic basic/end-to-end examples.
- Added visualization and Scientific UX tests and changed CI to install the `viz` extra so Matplotlib figures are exercised under Python 3.10 and 3.11.
- Bumped the package version to `0.7.0`; the stable `AgencityResult` serialization schema remains `0.3` and the structured analysis schema remains `0.5`.

### Unified-theory correction

- Corrected the scalar public API after re-reading Volume 2 of the same Theory of Agencity. Volume 2 keeps the CRM width `w > 0` distinct from characteristic structural time `tau` and only often chooses `w=tau` as a convention. `compute_agencity()` now uses `w=tau` when omitted and preserves an explicitly supplied positive `w` instead of rejecting `w != tau`.
- Updated CRM-dependent finite-record warm-up from the old `2*tau` implementation assumption to `2*w`. The separate derived indicator `Sigma_Theta(t) = Var(Theta(s); s in [t-tau,t])` remains based on `tau` exactly as stated in the complete formulary.
- Updated multiscale/window/discrete/multivariate terminology so Volume 2 definitions are described as part of the same theory, not as an alternative “advanced theory.”
- No formula for CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b` was modified to obtain the UX or test results.

## 0.6.0 - 2026-08-10

### Multiscale & Extensions

- Replaced the legacy multiscale implementation with an explicit extension engine that preserves the canonical scalar equations and never silently compresses the CRM window or infers physical parameters from signal statistics.
- Added a time-resolved `b(t, tau)` spectrum with explicit `tau` and `w` coordinates. The default extension convention is `w = tau` at every scale; independent `w` was initially exposed through the extension path.
- At the time of 0.6.0 the stable `compute_agencity()` contract still required `w = tau`; **0.7.0 supersedes that software restriction** after applying Volume 2's explicit distinction between `w` and `tau`.
- Added the Chapter 13 angular-stability window optimiser `Phi2(w)`, with discrete candidate windows `w = N delta` and explicit exclusion of candidates whose structural orientation is undefined because no complete interval has `S > 0`.
- Added `compute_discrete_agencity()` as a sampled-sequence convenience API that constructs a uniform coordinate and delegates to the existing scalar discrete implementation instead of duplicating equations.
- Added the Volume 2 multivariate construction: scalar Agencity is computed independently per component, `beta_multi` is the pointwise `P_c`-weighted average, and `b_total` is the vector-additive sum of component fluxes. Scalar, per-component, and sampled component-power inputs are supported.
- Reworked analysis-side multiscale helpers to require explicit `A_ref` and `P_c`, to reject physical normalization of `b`, and to label `find_optimal_tau()` as diagnostic selection across a supplied scale grid rather than inference of the physical characteristic time.
- Added `docs/multiscale_extensions.md` and dedicated extension tests covering spectrum/scalar equivalence, independent-window isolation, window optimisation, sampled-signal equivalence, `P_c`-weighted multivariate aggregation, time-varying component power, and the Riemannian implementation boundary.
- Added public APIs `compute_agencity_spectrum()`, `optimize_agencity_window()`, `compute_discrete_agencity()`, `compute_multivariate_agencity()`, and `riemannian_extension_status()`.
- Bumped the package version to `0.6.0`; the `AgencityResult` serialization schema remained unchanged.

### Scientific boundary

- `tau`, CRM window `w`, sampling interval `delta`, and a multiscale scan are separate concepts. A spectrum maximum is not automatically the physical `tau`, and an optimised signal-derived `w` is not silently promoted to the characteristic structural time.
- The multivariate construction follows the accepted `P_c`-weighted formula and additive flux law; it does not invent a coupled-vector CRM theory.
- Volume 2 sketches a Riemannian formulation but explicitly defers detailed analysis. v0.6 therefore did not invent a production Riemannian pipeline and reported that part as `experimental` and unimplemented.
- No v0.6 extension changed `beta`, `J`, CRM, `M`, `O`, `D`, `S`, `P_c`, `A_ref`, or the exact rest-state convention.

## 0.5.0 - 2026-08-10

### Agencity Analysis

- Established a strict analysis layer that consumes `AgencityResult` outputs without recomputing or redefining `A_ref`, `tau`, CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b`.
- Implemented the theory-derived local angular variance `Sigma_Theta(t) = Var(Theta(s); s in [t-tau,t])` on complete structurally valid windows. Circular resultant variance remains a separately labelled diagnostic rather than a replacement for `Sigma_Theta`.
- Reworked structural coherence to use `Theta = atan2(O, M)` and to exclude `S = 0` points where orientation is physically undefined. `arg(beta)`/`arg(b)` is not substituted for structural orientation because the sign of `J` can add a pi phase shift.
- Reworked the real-agencity criterion around `S > 0`, low `Sigma_Theta`, and significant `|b|` without universal thresholds. Local evaluation requires explicit contextual thresholds; a global Boolean additionally requires an explicit persistence fraction.
- Replaced the legacy epsilon-modified trajectory curvature with the signed algebraic curvature of the intrinsic `beta(t)` curve, leaving curvature undefined when `beta_dot = 0` instead of inserting epsilon into the denominator.
- Added structural winding diagnostics from `Theta`, including explicit undefined handling across structural zeros and raw finite-interval winding without forced integer quantisation.
- Added exact agencity-zero detection from `S = 0` or `J = 0`, critical-surface crossings `D = S`, and explicit-threshold wrapped `Theta` jump detection.
- Added local peak diagnostics for dynamic intensity `D` and explicit-threshold structural plateaus of `S`.
- Added threshold-free regime signatures and contextual `RegimeCriteria` classification for the theory table (`null`, `passive_damped`, `active_oscillating`, `unstable`, `stochastic`, `chaotic`). Non-null classification defaults to `undetermined` when criteria are not supplied.
- Made multiscale scaling signatures reject non-positive log inputs instead of replacing them by epsilon. Qualitative slope interpretation now requires an explicit diagnostic threshold.
- Added `analyze_coherence()`, `analyze_geometry()`, and `analyze_regime_signature()` to the public analysis API while preserving existing analysis entry points.
- Added `docs/agencity_analysis.md` and dedicated analysis tests for `Sigma_Theta`, real-agencity threshold policy, beta curvature, winding, exact zeros, transitions, D peaks, regime criteria, signatures, and non-mutation of computed arrays.
- Bumped the package version to `0.5.0`; the `AgencityResult` serialization schema remained `0.3` because the result model was unchanged.

### Scientific boundary

- Version 0.5.0 created scientific diagnostics from computed outputs; it did not modify the scalar equations and was not empirical confirmation of the theory.
- Numerical criteria for "low" angular variance, "significant" flow, peak prominence, structural plateaus, persistence fractions, and regime classification are contextual diagnostics, not universal constants.
- Noise and chaos may contain local non-zero `beta` and may satisfy a local real-agencity criterion intermittently; the API does not convert such a local event into a whole-record claim without an explicit persistence rule.
- Geometry is computed from intrinsic `beta`, not from `b`, so an externally varying `P_c(t)` cannot silently alter the state-trajectory curvature or winding analysis.

## 0.4.0 - 2026-08-10

### Scientific validation

- Added a deterministic scientific reference battery covering exact rest, a sinusoid, an underdamped passive oscillator, the Van der Pol oscillator, a negative-damping unstable oscillator, a low-pass-filtered Ornstein-Uhlenbeck process, and the classical Lorenz system.
- Added theory-facing regime checks for exact nullity, periodic structure, passive structure-dominated tails, bounded self-sustained oscillation, unstable logarithmic-contrast growth, and irregular stochastic/chaotic orientation.
- Added mathematical-property tests for state-translation invariance, global sign-inversion invariance, temporal covariance under simultaneous time/tau rescaling, small structured-amplitude behaviour, large-amplitude logarithmic growth, and exact linearity in characteristic power.
- Added a uniform-refinement convergence experiment against a fine-grid reference and a smooth-perturbation robustness experiment. These are numerical validation checks, not replacements for the analytical theorems.
- Added explicit finite-record CRM warm-up handling to validation metrics (`t >= t0 + 2*tau`) under the then-current `w=tau` convention; v0.7 generalises this software rule to `2*w` when `w` is explicit.
- Added `docs/scientific_validation.md`, distinguishing theorem-level claims, fixed-benchmark numerical observations, numerical tolerances, and empirical validation.
- Bumped the package version to `0.4.0` without changing the v0.2 scalar equations or v0.3 result schema.

### Scientific boundary

- Version 0.4.0 validates the implementation against selected consequences and reference regimes stated by the accepted theory; it is not experimental confirmation that Agencity is a universal physical observable.
- Benchmark tolerances are fixed numerical acceptance criteria, never universal thresholds for coherent or "real" agencity.
- Filtered noise is explicitly allowed to have non-zero `D` and local non-zero `beta`; stochastic validation focuses on reproducible structural/orientational behaviour rather than forcing a null signal.
- Current `e = exp(1)` governs v0.4 tests. Earlier numerical examples using a different effective offset convention are not silently converted into current acceptance targets.

## 0.3.0 - 2026-08-10

### Stable computational API

- Stabilized `compute_agencity()` as the scalar-signal reference entry point without changing the v0.2 equations.
- Added strict one-dimensional input validation, explicit ambiguity errors for `u` versus the compatibility alias `data`, and rejection of unknown compute keywords instead of silently ignoring them.
- Added typed public exceptions for validation, physical-parameter, unit-label, batch, and streaming failures while keeping validation exceptions compatible with existing `ValueError` handling.
- Added descriptive unit-label support: `unit` for `u`/`A_ref`, `coordinate_unit` for `xi`/`tau`, and `power_unit` for `P_c`; observable flux `b` is labelled as informational power (`power_unit·nat`, e.g. `W·nat`). No hidden unit conversion is performed.
- Stabilized `ExperimentMetadata` with validation, unit contracts, `memory_window`, forward-compatible unknown-field preservation, and explicit separation of legacy observational metadata from theoretical modifiers.
- Stabilized `AgencityResult` with schema version `0.3`, scalar or sampled strictly positive `P_c`, consistent metadata synchronization, complex round-tripping, wrapped `theta = angle(U)`, and exact `eta = |b| / P_c` without epsilon substitution.
- Restored explicit support for externally supplied time-varying `P_c(t)` as a sampled profile or callable, preserving `b(t) = P_c(t) beta(t)` without deriving power from the observed signal.
- Preserved compatibility fields and aliases: `data=`, historical `Pc=`, legacy serialized physical-field names, legacy `metadata.extra["memory_window"]`, and summary keys including `Pc_mean`, `A_fact`, and `resolution_scale`.
- Improved deserialization so legacy payloads may recover `A_ref`, `tau`, and scalar `P_c` from metadata before any compatibility default is considered.
- Improved batch execution with per-item physical parameters, metadata/config overrides, deterministic ordering, and indexed `BatchItemError` failures.
- Improved streaming with persistent physical context, continuous implicit coordinates across chunks, explicit coordinate-order validation, and `StreamNotReadyError` when there is not yet enough CRM history.
- Repaired fluent-pipeline compatibility so `set_tau()` and `set_power()` affect the physical metadata actually used by computation. `set_resolution_scale()` remains observational metadata only, and `set_activity_factor()` is deprecated metadata that does not modify CRM.
- Added a dedicated v0.3 stable-API test suite and user documentation.

### Scientific boundary

- Version 0.3.0 is an API-stability milestone, not a change to physics and not empirical validation of the Theory of Agencity.
- `beta`, `J`, CRM, `M`, `O`, `D`, `S`, `tau`, `w`, `P_c`, and `A_ref` are not redefined for software convenience.
- Time-varying `P_c(t)` support is an implementation of the multiplicative flux relation, not a signal-derived power estimator.

## 0.2.0 - 2026-08-10

### Canonical Core

- Reconciled the scalar reference pipeline with the current second-edition theory: `u -> u* -> X* -> A* -> M,O -> D,S -> J,U -> beta -> b`.
- Implemented exact normalization `u* = u / A_ref` and reduced time `t* = t / tau`.
- Corrected memory to `M = CRM[u*]` and organisation to the cross-correlation `O = CRM[u*, X*]`.
- Fixed the then-current project reference CRM window to `w = tau`; removed historical `tau / A_fact` compression. Volume 2's independent `w` is now honored by the public API as of 0.7.0.
- Removed historical `tanh` saturation from memory and organisation.
- Implemented `D = sqrt(X*^2 + (A* X*)^2)` and `S = sqrt(M^2 + O^2)` without clipping or saturation.
- Implemented `J = ln((e + D)/(e + S))` with the theoretical constant `e = exp(1)` and no machine epsilon inserted into the equation.
- Implemented `U = (M + iO)/S` for `S > 0`, with the explicit branch `U = beta = 0` for `S = 0`.
- Kept `b = P_c beta` exact and linear in characteristic power.
- Made `A_ref`, `tau`, and `P_c` explicit physical/contextual parameters: resolution uses explicit values, metadata, documented physical energetics, or deliberately registered conventions, never silent signal-statistical fallbacks.
- Added an exact null/rest-state precheck. An exactly constant sampled observable bypasses derivative and CRM evaluation and returns `X*=A*=M=O=D=S=J=U=beta=b=0` exactly, following the project maintainer's postulate interpretation rather than trying to prove rest numerically through finite differences.
- Kept numerical epsilon and generic safeguard helpers outside valid physical equations.

### Tests and traceability

- Added analytical tests for normalization, reduced derivatives, CRM auto/cross behaviour, zero-variance CRM, tiny non-zero signals, `M/O`, dynamic/structural norms, logarithmic contrast, orientation, `S = 0`, `beta`, and `b`.
- Added tiny-positive-value tests to ensure machine epsilon does not redefine physical zero.
- Added tests proving that the exact rest state bypasses derivative and CRM operators.
- Updated smoke, integration, regression, README, overview, and theory mapping for explicit physical parameters and the v0.2 pipeline.

### Scientific boundary

- Version 0.2.0 is an implementation-fidelity milestone, not empirical validation of the Theory of Agencity.
- Coherence, regimes, real-agencity diagnostics, multiscale optimisation, and speculative field/quantum/cosmological extensions remain separate from the scalar computational core.

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

The scientific reconciliation items deferred from 0.1.12 were resolved in 0.2.0 according to the current second-edition theory. Historical Git formulations are not used to define physics.

## 0.1.0

- Initial project skeleton.