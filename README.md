# AgencityLab

AgencityLab is an alpha-stage scientific Python project for implementing and experimenting with the Theory of Agencity developed by Gilbert BEMWIZ.

The project is research software. Its purpose is to make the theory inspectable, testable, reproducible, and progressively comparable with data. It is not currently evidence that Agencity is a universally validated physical observable, a classifier of agency, or a replacement for established statistical measures.

## Status of the implementation

Version `0.8.0` combines the scalar Theory of Agencity engine, stable computational API, scientific-validation battery, analysis layer, multiscale/discrete/multivariate constructions, researcher-facing Scientific UX, and a measured engineering/performance layer. Version 0.8 optimizes implementation details and packaging; it does not redefine the accepted theory.

```text
u -> u* -> X* -> A* -> M, O -> D, S -> J, U -> beta -> b
```

The scalar construction remains:

```text
u* = u / A_ref
t* = t / tau
X* = d(u*) / d(t*)
A* = d^2(u*) / d(t*)^2
M = CRM[u*]              with CRM width w > 0
O = CRM[u*, X*]
D = sqrt((X*)^2 + (A* X*)^2)
S = sqrt(M^2 + O^2)
J = ln((e + D) / (e + S))
U = (M + i O) / S       for S > 0, else 0
beta = J U               for S > 0, else 0
b(t) = P_c(t) beta(t)
```

Volume 2 keeps `w` distinct from `tau` and often uses `w=tau` as a convenient convention. AgencityLab follows that rule: omitting `w` uses `w=tau`; explicitly supplying a positive `w` preserves it. Historical `tanh` saturation, `tau / A_fact` CRM compression, signal-derived physical fallbacks, and epsilon-modified physical denominators are not used by the reference pipeline. An exactly constant sampled observable is treated as the project null/rest-state convention and bypasses derivative/CRM evaluation exactly.

## Installation

The minimal package requires only NumPy:

```bash
pip install agencitylab
```

Install only the optional capability needed by the study:

```bash
pip install "agencitylab[scientific]"  # SciPy reference-system generators
pip install "agencitylab[data]"        # pandas and xarray result adapters
pip install "agencitylab[viz]"         # matplotlib figures
pip install "agencitylab[export]"      # Excel and PDF exports
pip install "agencitylab[numba]"       # experimental Numba primitives
pip install "agencitylab[jax]"         # experimental JAX primitives
```

`ml` remains as a compatibility alias for the combined Numba/JAX extra. New environments should prefer the narrower `numba` or `jax` extra. Development installation is:

```bash
python -m pip install -e ".[dev,viz]"
```

The core import does not require SciPy, pandas, xarray, matplotlib, Numba, or JAX. CI builds both wheel and source distribution, installs each into a fresh virtual environment, runs a minimal canonical computation, and tests supported extras separately.

## Minimal example

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

print(result.b.shape)
print(result.b_unit)
print(result.summary())
```

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual inputs. `compute_agencity()` does not infer them from standard deviation, MAD, range, z-score, or arbitrary signal statistics. Chapter 13 provides separate theory-defined procedures for selecting `w` from data when that is the scientific question.

The public API also accepts an externally specified sampled or callable `P_c(t)`. Such a profile must match the computation coordinate and is never derived from `u`.

Unit arguments are descriptive labels only: AgencityLab does not silently convert magnitudes between unit systems. `unit` applies to `u` and `A_ref`, `coordinate_unit` to `xi`, `tau`, and `w`, and `power_unit` to `P_c`. The observable `b` is labelled with the corresponding informational-power unit, for example `W·nat` when `P_c` is in watts.

## Engineering and performance in 0.8

The pre-v0.8 CRM recomputed two means, two variances, and one covariance for every sample in a Python loop, giving `O(N*w)` work. The reference NumPy implementation now uses rolling first and second moments for the ordinary path, reducing it to `O(N)` in signal length. Exactly constant windows are detected without a tolerance, short or numerically delicate windows use the direct centred Pearson formula, and no epsilon is inserted into a physical denominator.

The repository includes:

- direct before/after CRM and complete-pipeline benchmarks;
- numerical-equivalence checks against the retained pre-v0.8 reference;
- a 100,000-sample long-signal test;
- per-item batch-physics and serial/thread equivalence tests;
- full-history streaming versus one-shot equivalence;
- multiscale rows versus independent canonical calculations;
- approximate peak-memory observations;
- isolated wheel, source-distribution, and optional-extra installation checks.

Run the CI-sized benchmark with:

```bash
python benchmarks/performance/benchmark_v08.py --quick
```

Timing values are observations, not CI thresholds. Scientific equivalence is tested separately. See [`docs/engineering_performance.md`](docs/engineering_performance.md) and [`benchmarks/README.md`](benchmarks/README.md).

### Backend status

- **NumPy — stable:** reference numerical engine and the complete canonical public pipeline.
- **Numba — experimental:** optional one-dimensional primitives; it does not silently replace the canonical pipeline.
- **JAX — experimental:** optional autodiff/vectorisation primitives, normally float32 unless JAX x64 is enabled; it does not silently replace the canonical pipeline.

`backend_capabilities()` reports availability, stability, scope, default precision, and whether a backend owns the canonical pipeline. A requested optional backend is recorded in result configuration, while `canonical_backend` remains `numpy`.

## Scientific UX introduced in 0.7

The direct research workflow is:

```text
signal -> result -> diagnostic -> report -> figure -> export
```

A complete reproducible example is:

```python
from agencitylab import scientific_workflow

study = scientific_workflow(
    u,
    xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
    coordinate_unit="s",
    power_unit="W",
    export_dir="agencity_output",
    show=False,
)

print(study.report)
print(study.analysis["real_agencity"]["status"])
print(study.exports)
```

`ScientificStudy` keeps `result`, `analysis`, `report`, `figures`, and `exports` separate. The workflow is orchestration only: diagnostic thresholds never modify the computed theoretical arrays.

Research-facing visualizations include `overview`, `geometry`, `diagnostics`, explicit complex time-series/component/frequency compatibility views, and the multiscale `b(t,tau)` spectrum. CSV contains one row per sample with explicit real, imaginary, and magnitude columns; JSON preserves the stable result serialization, structured analysis, and optional text report.

See [`docs/scientific_ux.md`](docs/scientific_ux.md), [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md), and [`docs/tutorials/full_pipeline.md`](docs/tutorials/full_pipeline.md).

## Stable public API

`compute_agencity()` accepts one finite one-dimensional scalar observable. `data=` remains a compatibility alias for `u=`, but passing both is an explicit error. Unknown keywords are rejected rather than silently ignored.

`AgencityResult` validates its numerical payload, supports scalar or sampled `P_c`, keeps wrapped `theta = angle(U)` rather than silently unwrapping phase, uses the stable `0.3` serialization schema, and exposes unit metadata. `ExperimentMetadata` preserves unknown fields for forward compatibility and keeps physical/contextual parameters separate from signal-derived quantities.

Batch items can carry independent `A_ref`, `tau`, `w`, `P_c`, configuration, and metadata. Results preserve input order in serial and parallel modes. Full-history streaming maintains monotonically increasing coordinates and produces the same final result as one-shot computation; an explicit finite `window_size` is an implementation choice that intentionally limits retained history and is never applied implicitly.

See [`docs/stable_api.md`](docs/stable_api.md).

## Scientific validation introduced in 0.4

The reference bench covers seven fixed systems: exact rest, sinusoid, underdamped oscillator, Van der Pol oscillator, negative-damping unstable oscillator, low-pass-filtered Ornstein-Uhlenbeck process, and the classical Lorenz system. It checks stated theory consequences without tuning equations to the resulting numbers.

It also tests translation and sign-inversion invariance, temporal covariance when time and `tau` are scaled together, small- and large-amplitude limits, exact `P_c` linearity, uniform-grid convergence, and robustness to decreasing smooth perturbations.

A green scientific-validation suite means that the implementation reproduces the tested mathematical and numerical consequences under the documented benchmark conditions. It is not empirical confirmation of the Theory of Agencity itself. See [`docs/scientific_validation.md`](docs/scientific_validation.md).

## Agencity Analysis introduced in 0.5

The analysis layer transforms a computed result into diagnostics without changing canonical arrays. It implements structural angular variance and coherence, the contextual real-agencity diagnostic, signed curvature of the `beta` trajectory, winding diagnostics, zeros and critical-surface crossings, events and transitions, threshold-free finite-record signatures, contextual regime classification, and multiscale signature fits.

Conservative defaults are intentional:

```python
from agencitylab import analyze_agencity

analysis = analyze_agencity(result)
print(analysis["regime"])
print(analysis["real_agencity"]["status"])
```

For non-null data, interpretations remain `undetermined` by default when required contextual criteria are absent. A single intermittent sample in noise or chaos is not enough to make a whole-record real-agencity claim without an explicit persistence rule. See [`docs/agencity_analysis.md`](docs/agencity_analysis.md).

## Multiscale, window, discrete, and multivariate constructions

- `compute_agencity_spectrum()` returns the time-resolved `b(t,tau)` spectrum plus scalar summaries. By default every scale uses `w=tau`; `windows=` keeps `w` explicit and independent.
- `optimize_agencity_window()` implements the Chapter 13 angular-stability criterion `Phi2`, searching discrete windows `w=N delta` without treating undefined structural orientation as artificial zero variance.
- `compute_discrete_agencity()` constructs a uniform coordinate and delegates to the scalar implementation.
- `compute_multivariate_agencity()` computes scalar Agencity component by component, then forms the theory-specified pointwise `P_c`-weighted state and additive total flux.
- `riemannian_extension_status()` reports the Riemannian construction as experimental and not implemented because Volume 2 defers the detailed analysis needed to fully specify and test it.

`tau`, `w`, sampling interval `delta`, and a multiscale scan are different objects. A peak in a scale spectrum is a diagnostic result; it is not silently promoted to the physical characteristic time. See [`docs/multiscale_extensions.md`](docs/multiscale_extensions.md).

## Repository map

- `agencitylab/core/`: deterministic mathematical operators and theory-defined computation tools; no plotting or regime interpretation.
- `agencitylab/api/`: stable user-facing orchestration for compute, analysis, extensions, exports, visualizations, batch, streaming, and scientific workflows.
- `agencitylab/analysis/`: diagnostics, geometry, coherence, events, transitions, signatures, classifications, and reports.
- `agencitylab/backends/`: reference NumPy primitives plus explicitly experimental optional primitives.
- `agencitylab/visualization/`: presentation of already-computed results and diagnostics.
- `agencitylab/models/`: reproducibility-oriented result and metadata containers.
- `tests/`: analytical, API, scientific-validation, analysis, extension, visualization, integration, regression, packaging, and foundation tests.
- `docs/`: theory mapping, API, validation, analysis, extensions, UX, engineering, examples, and tutorials.
- `benchmarks/scientific/`: deterministic theory-facing reference systems.
- `benchmarks/performance/`: reproducible implementation benchmarks and retained comparison reference.

## Documentation

Start with:

- [`docs/engineering_performance.md`](docs/engineering_performance.md) for the v0.8 audit, measured benchmarks, memory, packaging, and backend scope.
- [`docs/scientific_ux.md`](docs/scientific_ux.md) for researcher workflows, figures, exports, and reproducibility.
- [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md) for a short signal-to-export example.
- [`docs/tutorials/full_pipeline.md`](docs/tutorials/full_pipeline.md) for a complete staged workflow.
- [`docs/examples/visualization_gallery.md`](docs/examples/visualization_gallery.md) for the figure API.
- [`docs/multiscale_extensions.md`](docs/multiscale_extensions.md) for multiscale, `w`, discrete, multivariate, and Riemannian boundaries.
- [`docs/agencity_analysis.md`](docs/agencity_analysis.md) for diagnostics.
- [`docs/scientific_validation.md`](docs/scientific_validation.md) for validation scope and reference systems.
- [`docs/stable_api.md`](docs/stable_api.md) for the public computational contract.
- [`docs/theory_mapping.md`](docs/theory_mapping.md) for definitions, parameter policy, null-state convention, and numerical boundaries.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and scientific-change rules.

## Development checks

The repository supports Python 3.10, 3.11, and 3.12:

```bash
python -c "import agencitylab; print(agencitylab.__version__)"
ruff check agencitylab tests benchmarks/performance
pytest
python -m build
python benchmarks/performance/benchmark_v08.py --quick
```

CI also performs clean wheel/sdist installation and isolated optional-extra smoke tests. The Ruff policy remains a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`) so scientific changes are not obscured by unrelated style churn.

## Scientific caution

Agencity is an emerging theoretical framework. Implementation fidelity, API stability, deterministic reference validation, diagnostic tooling, and theory-defined extensions are not empirical validation. High dynamic intensity is not, by itself, evidence of agency, and `beta != 0` does not establish coherent or “real” agencity.

Analysis thresholds, peak filters, plateau tolerances, persistence fractions, regime criteria, and scale selections are diagnostics. When Chapter 13 selects an optimal `w` from data, the result should be recorded as a theory-defined signal-derived window selection rather than silently confused with `tau`.

Experimental, heuristic, diagnostic, or legacy components remain labelled as such. The Riemannian pipeline is experimental specifically because the theory document itself defers its detailed analysis.

## Author and repository lineage

Theory and original project: **Gilbert BEMWIZ**.

Active integration repository: `somafgroup/AgencityLab`.

Original repository lineage: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
