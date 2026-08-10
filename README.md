# AgencityLab

AgencityLab is an alpha-stage scientific Python project for implementing and experimenting with the Theory of Agencity developed by Gilbert BEMWIZ.

The project is research software. Its purpose is to make the theory inspectable, testable, reproducible, and progressively comparable with data. It is not currently evidence that Agencity is a universally validated physical observable, a classifier of agency, or a replacement for established statistical measures.

## Status of the implementation

Version `0.7.0` combines the scalar Theory of Agencity engine, stable computational API, scientific-validation battery, analysis layer, multiscale/discrete/multivariate constructions, and a researcher-facing Scientific UX. The two theory documents are treated as two volumes of the **same theory**; when Volume 2 specifies or generalises a construction, it governs the implementation.

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

Volume 2 keeps `w` distinct from `tau` and often uses `w=tau` as a convenient convention. AgencityLab now follows that rule: omitting `w` uses `w=tau`; explicitly supplying a positive `w` preserves it. Historical `tanh` saturation, `tau / A_fact` CRM compression, signal-derived physical fallbacks, and epsilon-modified physical denominators are not used by the reference pipeline. An exactly constant sampled observable is still treated as the project null/rest-state convention and bypasses derivative/CRM evaluation exactly.

## Installation

Core package:

```bash
pip install agencitylab
```

Scientific visualizations:

```bash
pip install "agencitylab[viz]"
```

From a source checkout for development:

```bash
python -m pip install -e ".[dev,viz]"
```

Optional feature groups include `viz`, `app`, `ml`, `export`, and `docs`.

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

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual inputs to the computation. `compute_agencity()` does not infer them from standard deviation, MAD, range, z-score, or arbitrary signal statistics. Chapter 13 provides separate theory-defined procedures for selecting `w` from data when that is the scientific question.

The public API also accepts an externally specified sampled or callable `P_c(t)` when a time-dependent characteristic-power context is required. Such a profile must match the computation coordinate and is never derived from `u`.

Unit arguments are descriptive labels only: AgencityLab does not silently convert magnitudes between unit systems. `unit` applies to `u` and `A_ref`, `coordinate_unit` to `xi`, `tau`, and `w`, and `power_unit` to `P_c`. The observable `b` is labelled with the corresponding informational-power unit, e.g. `W·nat` when `P_c` is in watts.

## Scientific UX in 0.7

Version 0.7 provides the direct research workflow:

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

Research-facing visualizations include:

- `visualize_agencity(result, kind="overview")` for the signal-to-flux pipeline;
- `kind="geometry"` for the intrinsic complex `beta` trajectory and signed curvature;
- `kind="diagnostics"` for `S`, `Sigma_Theta`, `|b|`, and dynamic/structural diagnostics;
- explicit complex time-series/component/frequency compatibility views;
- `visualize_multiscale_spectrum()` for the theoretical `b(t,tau)` scale spectrum.

CSV and JSON exports are designed for downstream research rather than presentation-only output:

```python
from agencitylab import export_result_csv, export_study_json

export_result_csv(result, "result.csv")
export_study_json(result, analysis, "study.json", text_report=report)
```

The CSV contains one row per sample with explicit real/imaginary/magnitude columns for complex quantities. The JSON bundle preserves the stable result serialization, structured analysis, and optional text report.

See [`docs/scientific_ux.md`](docs/scientific_ux.md), [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md), and [`docs/tutorials/full_pipeline.md`](docs/tutorials/full_pipeline.md).

## Stable public API

`compute_agencity()` accepts one finite one-dimensional scalar observable. `data=` remains a compatibility alias for `u=`, but passing both is an explicit error. Unknown keywords are rejected rather than silently ignored.

`AgencityResult` validates its numerical payload, supports scalar or sampled `P_c`, keeps wrapped `theta = angle(U)` rather than silently unwrapping phase, uses the stable `0.3` serialization schema, and exposes unit metadata. `ExperimentMetadata` preserves unknown fields for forward compatibility and keeps physical/contextual parameters separate from signal-derived quantities.

Batch items can carry per-item physical parameters and metadata. Streaming maintains monotonically increasing implicit coordinates across chunks and raises an explicit `StreamNotReadyError` when there is not yet enough CRM history.

See [`docs/stable_api.md`](docs/stable_api.md) for the computational API contract.

## Scientific validation in 0.4

The v0.4 reference bench covers seven fixed systems: exact rest, sinusoid, underdamped oscillator, Van der Pol oscillator, negative-damping unstable oscillator, low-pass-filtered Ornstein-Uhlenbeck process, and the classical Lorenz system. The suite checks stated theory consequences without tuning the equations to the resulting numbers.

It also tests translation and sign-inversion invariance, temporal covariance when time and `tau` are scaled together, small- and large-amplitude limits, exact `P_c` linearity, uniform-grid convergence, and robustness to decreasing smooth perturbations.

A green scientific-validation suite means that the implementation reproduces the tested mathematical/numerical consequences under the documented benchmark conditions. It is not empirical confirmation of the Theory of Agencity itself. See [`docs/scientific_validation.md`](docs/scientific_validation.md).

## Agencity Analysis in 0.5

The analysis layer transforms a computed result into scientific diagnostics without changing its arrays. It implements:

- theoretical local angular variance `Sigma_Theta(t) = Var(Theta(s); s in [t-tau,t])` on complete structurally valid windows;
- structural phase coherence based on theoretical `Theta`, not on `arg(beta)` or `arg(b)`;
- the real-agencity criterion `S > 0`, low `Sigma_Theta`, significant `|b|`, with no universal numerical thresholds;
- signed algebraic curvature of the **beta trajectory**, with no epsilon in the curvature denominator;
- structural winding number diagnostics and explicit undefined handling across `S = 0`;
- exact agencity zeros from `S = 0` or `J = 0`, and critical-surface crossings `D = S`;
- explicit-threshold orientation jumps, local peaks of `D`, and explicit-threshold plateaus of `S`;
- threshold-free finite-record regime signatures;
- contextual classification into null, passive damped, active oscillating, unstable, stochastic, or chaotic regimes through explicit `RegimeCriteria`;
- multiscale signature fits without epsilon substitution or hidden slope thresholds.

Conservative defaults are intentional:

```python
from agencitylab import analyze_agencity

analysis = analyze_agencity(result)
print(analysis["regime"])
print(analysis["real_agencity"]["status"])
```

For non-null data, these interpretations remain `undetermined` by default when the required contextual criteria are absent. A single intermittent sample in noise or chaos is not enough to make a whole-record real-agencity claim without an explicit persistence rule.

See [`docs/agencity_analysis.md`](docs/agencity_analysis.md).

## Multiscale, window, discrete, and multivariate constructions

- `compute_agencity_spectrum()` returns the time-resolved `b(t,tau)` spectrum plus scalar summaries. By default every scale uses the common `w=tau` convention; `windows=` keeps `w` explicit and independent.
- `optimize_agencity_window()` implements the Chapter 13 angular-stability criterion `Phi2`, searching discrete windows `w=N delta` without treating undefined structural orientation as artificial zero variance.
- `compute_discrete_agencity()` is a convenience entry point for uniformly sampled sequences and delegates to the same scalar implementation.
- `compute_multivariate_agencity()` computes scalar Agencity component by component, then forms the theory-specified pointwise `P_c`-weighted state and additive total flux.
- `riemannian_extension_status()` reports the Riemannian construction as **experimental and not implemented** because Volume 2 explicitly defers the detailed analysis needed to fully specify and test a production pipeline.

`tau`, `w`, sampling interval `delta`, and a multiscale scan are different objects. A peak in a scale spectrum is a diagnostic result; it is not silently promoted to the physical characteristic time.

See [`docs/multiscale_extensions.md`](docs/multiscale_extensions.md).

## Repository map

- `agencitylab/core/`: deterministic mathematical operators and theory-defined computation tools. No plotting or regime interpretation belongs here.
- `agencitylab/api/`: stable user-facing orchestration, including compute, analysis, extensions, exports, visualizations, and the v0.7 scientific workflow.
- `agencitylab/analysis/`: derived indicators, diagnostics, geometry, coherence, events, transitions, signatures, classifications, and reports.
- `agencitylab/visualization/`: scientific presentation of already-computed results and diagnostics.
- `agencitylab/models/`: reproducibility-oriented result and metadata containers.
- `tests/`: analytical, API, scientific-validation, analysis, extension, visualization, integration, regression, and foundation tests.
- `docs/`: theory mapping, stable API, validation, analysis, extensions, Scientific UX, examples, and tutorials.
- `benchmarks/scientific/`: deterministic theory-facing reference systems.

## Documentation

Start with:

- [`docs/scientific_ux.md`](docs/scientific_ux.md) for v0.7 researcher workflows, figures, exports, and reproducibility.
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

The repository supports Python 3.10 and 3.11. CI verifies both versions with visualization support installed:

```bash
python -c "import agencitylab; print(agencitylab.__version__)"
ruff check agencitylab tests
pytest
python -m build
```

The Ruff policy remains a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`) so scientific changes are not obscured by unrelated style churn.

## Scientific caution

Agencity is an emerging theoretical framework. Implementation fidelity, API stability, deterministic reference validation, diagnostic tooling, and theory-defined extensions are not empirical validation. High dynamic intensity is not, by itself, evidence of agency, and `beta != 0` does not establish coherent or “real” agencity.

Analysis thresholds, peak filters, plateau tolerances, persistence fractions, regime criteria, and scale selections are diagnostics. When Chapter 13 selects an optimal `w` from data, the result should be recorded as a theory-defined signal-derived window selection rather than silently confused with `tau`.

Experimental, heuristic, diagnostic, or legacy components remain labelled as such. The Riemannian pipeline is currently experimental specifically because the theory document itself defers its detailed analysis.

## Author and upstream

Theory and original project: **Gilbert BEMWIZ**.

Canonical upstream repository: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
