# AgencityLab

AgencityLab is an open-source Python framework implementing and testing the **Theory of Agencity**.

**Current software status: 1.1.4.** Version 1.0 freezes the documented stable scalar software contract. The 1.1.x line adds explicitly labelled experimental, research, and speculative capabilities without redefining the canonical observable. Software stability is distinct from empirical validation of the theory.

## Canonical observable

The central observable is

```text
b(t) = P_c(t) * beta(t)
```

with the canonical pipeline

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,Theta -> beta -> b
```

and

```text
S = sqrt(M^2 + O^2)
Theta = atan2(O, M)
J = ln((e + D) / (e + S)),  e = exp(1)
```

For `S > 0`:

```text
U = (M + i O) / S
beta = J * U
b = P_c * beta
```

For `S = 0`, the canonical convention is `U = 0` and `beta = 0`. `P_c` is finite and non-negative; `P_c = 0` gives `b = 0` exactly. Numerical epsilon is not inserted into these valid equations.

## Installation

The stable canonical engine and default public API require only NumPy:

```bash
pip install agencitylab
```

Optional capabilities are isolated:

```bash
pip install "agencitylab[scientific]"
pip install "agencitylab[data]"
pip install "agencitylab[viz]"
pip install "agencitylab[export]"
pip install "agencitylab[docs]"
pip install "agencitylab[numba]"  # experimental
pip install "agencitylab[jax]"    # experimental
```

Numba and JAX do not replace the complete stable NumPy reference pipeline.

## Quickstart

```python
import numpy as np
import agencitylab

xi = np.linspace(0.0, 20.0, 801)
u = np.sin(xi)

result = agencitylab.compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
    coordinate_unit="s",
    power_unit="W",
)

print(result.b)
print(result.metadata.agencitylab_version)
```

`compute_agencity()` is the sole reference canonical end-to-end pipeline. `A_ref`, `tau`, `w`, and `P_c` are physical/contextual inputs. The stable compute path does not silently infer them from ordinary signal statistics. `tau` and CRM width `w` are distinct. If `w` is omitted, AgencityLab records and uses the implementation fallback `w=tau`; this is not a universal identity.

Diagnostics are separate:

```python
analysis = agencitylab.analyze_agencity(result)
print(analysis["real_agencity"]["status"])
```

For the complete visualisation/export walkthrough, install `agencitylab[viz,export]` and see `docs/tutorials/quickstart.md`.

## Observable and autonomous spatial fields

Version 1.1 adds an explicit field stack without changing the scalar core.

`compute_agencity_field()` applies the canonical temporal pipeline independently at each spatial location and returns `beta_obs(x,t)` and `b_obs(x,t)`. This orchestration has scientific status **experimental**; CRM remains temporal only.

Promotion to the autonomous research field is explicit:

```text
phi = sqrt(P_c * tau) * beta_obs
```

via `phi_from_observable_field()` or `beta_to_phi()`. `compute_agencity_field()` never performs this promotion automatically.

Version 1.1.2 exposes the research classical-field layer built on the shared quartic potential and NumPy numerical infrastructure:

- `simulate_klein_gordon()`;
- `simulate_dissipative_klein_gordon()`;
- `simulate_tdgl()`;
- `domain_wall_profile()` / `domain_wall_residual()`;
- `vortex_field()` / `vortex_radial_residual()`;
- `phase_winding()` and `field_zero_mask()`.

The autonomous field, its PDE dynamics, and coherent-structure references are **research** interfaces. Their implementation and numerical tests are not empirical confirmation of the underlying field-theory extension.

## Thermodynamics and classical gravity

Version 1.1.3 integrates two additional **research** layers on the same shared field contracts.

Thermodynamics provides source-defined evaluators for dissipation `Gamma |phi_dot|^2`, entropy production, temperature-dependent `lambda(T)`, distinct field and contrast agencial entropies, energy-balance and second-law residuals, Modulus Law and Phase Law evaluation, plus the explicitly conditional Volume-1 Landauer relations. The reported Phase-Law fit is available only as a named empirical reference; it is not a universal constant set.

Classical gravity provides limited Chapter-19/23 primitives: Minkowski geometry, matter/action densities, minimal stress-energy, Einstein-equation residuals, nonminimal action/field coupling, and an optional externally supplied U(1) gauge potential. It does **not** provide an Einstein solver, autonomous gauge dynamics, or a nonminimal stress-energy tensor when the accepted source does not fully specify one.

The gravity source convention is explicitly `(-,+,+,+)`, while the Chapter-16 flat dynamics remain `(+,-,-,-)`. AgencityLab preserves this source-level convention difference rather than silently changing signs to force equality.

Selected principal research APIs are available from the top-level package, while the full contracts remain organized under `agencitylab.thermodynamics` and `agencitylab.gravity`.

## Quantum and cosmological extensions

Version 1.1.4 adds two explicitly **speculative** layers. They implement source formulas for computational study; they are not experimental evidence for quantum Agencity, agentons, inflation, or dark energy.

`agencitylab.quantum` provides the Chapter-21 broken-symmetry radial/Goldstone masses and dispersions, explicit propagators with caller-supplied regulator, finite Fock-space approximations with the truncation defect exposed, the constant-bridge Agencity uncertainty bound, and only the explicitly stated leading one-loop quartic beta-function term. It does not provide generic path integrals, a vacuum-energy prescription, or a scattering engine.

`agencitylab.applications.cosmology` provides the homogeneous flat-FLRW field energy density and pressure, Friedmann residuals, and a deterministic RK4 reference solver. The first Friedmann equation is retained as a numerical constraint diagnostic rather than silently projected after every step. The negative minimum of the minimal broken quartic potential is preserved; no positive vacuum offset or cosmological constant is invented to make it resemble observed dark energy.

Selected principal speculative APIs are available at package top level, while the complete contracts remain under `agencitylab.quantum` and `agencitylab.applications.cosmology`.

## Stable v1.0 API

The principal stable scalar interfaces are:

- `compute_agencity()`;
- `AgencityResult`, `ExperimentMetadata`;
- `analyze_agencity()` and documented named diagnostics;
- `run_batch()`, `analyze_batch()`;
- `AgencityStream`, `stream_agencity()`;
- `compute_agencity_spectrum()`;
- `compute_discrete_agencity()`;
- `compute_multivariate_agencity()`;
- documented exports and visualisations;
- `ScientificStudy`, `scientific_workflow()`;
- `AgencityPipeline`, `pipeline()`.

The exact boundary is documented in `docs/stable_api.md`.

`compute_discrete_agencity()` is not an alias for the continuous sampled gradient chain. It implements the explicit Volume-2 centred first and second differences, with documented one-sided endpoint conventions. The mathematical/numerical distinction is described in `docs/theory_mapping.md`.

### Semantic Versioning and research APIs

Starting with 1.0.0, intentional breaking changes to the stable scalar public contract require a major version. Backwards-compatible stable functionality normally belongs to a minor release and stable bug fixes to patch releases.

The 1.1.x development line deliberately contains **experimental/research/speculative interfaces** that are outside the frozen v1.0 canonical stability guarantee. Those labelled interfaces may be added or refined within 1.1.x patch releases while the stable canonical scalar contract remains protected. Their scientific status must remain explicit.

## Canonical, diagnostic, experimental, research

**Stable canonical computation.** `agencitylab/core/` contains deterministic mathematical operators; normal user workflows use `compute_agencity()` as the unique reference orchestration. NumPy is the stable complete backend.

**Diagnostic analysis.** `agencitylab/analysis/` consumes computed results. Coherence, angular variance, real-agencity criteria, curvature, winding, events, transitions, signatures, regimes, and reports do not redefine the canonical state. In particular, `beta != 0` is not the definition of coherent or real agencity. Contextual structural, angular-stability, `|b|`, and persistence criteria remain diagnostic inputs and never modify `beta`.

Historical coherence/real-agencity helpers under `agencitylab.core` remain only as deprecated legacy diagnostics. Historical `compute_full_agencity()` is a compatibility wrapper around the public reference pipeline, not a second canonical implementation.

**Experimental.** Observable spatial orchestration, generic field numerics, Numba/JAX primitive layers, and signal-derived window optimisation are explicitly outside the stable canonical contract.

**Research / speculative.** Autonomous `phi` dynamics, coherent structures, thermodynamic and gravitational extensions are research layers. Quantum/agenton and cosmological extensions are speculative.

## Reproducibility

`AgencityResult` preserves the coordinate, observable, canonical intermediate arrays, physical/contextual parameters, unit/context labels, backend information, and producing AgencityLab version. Complex `beta` and `b` are preserved by JSON serialization; stable CSV export exposes real and imaginary components explicitly.

Research field solutions preserve spatial axes, model parameters, parameter provenance, numerical method, boundary handling, scientific status, and units convention through `DynamicalAgencityFieldSolution`. `FlatFLRWSolution` records its numerical trajectory and Friedmann constraint residual for reproducible speculative cosmology experiments.

This metadata supports traceability but does not make the inverse problem injective: the original observable cannot in general be reconstructed uniquely from `b` alone.

## Batch, streaming and extensions

Batch computations preserve input order and allow independent physical parameters per item. Supported serial/threaded execution is tested for scientific equivalence.

Streaming is retained-history recomputation. With full history its final result is tested against one-shot computation. A finite `window_size` intentionally changes the retained-history problem; the current implementation does not claim a constant-memory online recurrence.

Multiscale analysis scans explicit `tau` values and may use independent `w` values. It does not conflate characteristic time, CRM memory width, sampling interval, or physical parameter estimation. Discrete and multivariate constructions are part of the documented stable computational API; the incomplete Riemannian research extension is not promoted to stable status.

## Scientific validation

The deterministic regression suite covers exact rest, sinusoidal structure, passive damping, Van der Pol oscillation, negative-damping instability, filtered Ornstein-Uhlenbeck dynamics, and Lorenz dynamics. It also checks canonical identities, invariances/limits, CRM equivalence, edge cases, batch/streaming/multiscale consistency, discrete stencils and convergence, packaging, public workflows, classical field equations, coherent-structure references, thermodynamic identities/evaluators, gravity tensor/equation contracts, quantum primitive identities, homogeneous FLRW contracts, and cross-layer field contracts.

These checks validate implementation and numerical behaviour against accepted reference consequences; they are not universal empirical validation of the Theory of Agencity or evidence for speculative extensions.

## Documentation

- `docs/overview.md`
- `docs/stable_api.md`
- `docs/observable_fields.md`
- `docs/dynamical_field_foundations.md`
- `docs/classical_field_dynamics.md`
- `docs/coherent_structures.md`
- `docs/thermodynamics.md`
- `docs/gravity.md`
- `docs/quantum.md`
- `docs/cosmology.md`
- `docs/tutorials/quickstart.md`
- `docs/tutorials/full_pipeline.md`
- `docs/agencity_analysis.md`
- `docs/multiscale_extensions.md`
- `docs/theory_mapping.md`
- `docs/scientific_validation.md`
- `docs/engineering_performance.md`
- `docs/release_readiness.md`

Documentation is built in CI with significant Sphinx warnings treated as errors.

## Limitations

AgencityLab remains research software beyond its stable scalar computation contract. Results depend on the observable and physically/contextually justified parameters. Sampling and preprocessing decisions must be explicit. Sensitivity to `w` is scientifically meaningful. The inverse problem is non-injective. Current streaming is not constant-memory. Accelerated backends are experimental. Autonomous-field, thermodynamic, and gravitational extensions retain research status; quantum and cosmological extensions remain speculative. Gravity is not a general numerical-relativity framework, and the cosmology package is not an observational inference framework.

## Development and contribution

Supported Python versions are **3.10, 3.11, and 3.12**.

```bash
python -m pip install -e ".[dev,docs]"
ruff check agencitylab tests benchmarks/performance examples
pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m build
```

See `CONTRIBUTING.md` for scientific-change rules, branch/PR workflow, tests, and versioning policy.

## Citation

Scientific users should cite the software metadata in `CITATION.cff`. No DOI or affiliation is asserted unless it is provided by the project.

## License

AgencityLab is distributed under the MIT License. See `LICENSE`.
