# AgencityLab

AgencityLab is an alpha-stage scientific Python project for implementing and experimenting with the Theory of Agencity developed by Gilbert BEMWIZ.

The project is research software. Its purpose is to make the theory inspectable, testable, reproducible, and progressively comparable with data. It is not currently evidence that Agencity is a universally validated physical observable, a classifier of agency, or a replacement for established statistical measures.

## Status of the implementation

Version `0.6.0` keeps the v0.2 canonical scalar-signal `u -> b` core, the v0.3 stable computational API, the v0.4 scientific-validation battery, and the v0.5 Agencity Analysis layer, then adds explicit multiscale, independent-window, discrete, and multivariate extensions. The stable scalar reference path is unchanged and still uses `w = tau`.

```text
u -> u* -> X* -> A* -> M, O -> D, S -> J, U -> beta -> b
```

The canonical path remains:

```text
u* = u / A_ref
t* = t / tau
X* = d(u*) / d(t*)
A* = d^2(u*) / d(t*)^2
M = CRM[u*]
O = CRM[u*, X*]
D = sqrt((X*)^2 + (A* X*)^2)
S = sqrt(M^2 + O^2)
J = ln((e + D) / (e + S))
U = (M + i O) / S       for S > 0, else 0
beta = J U               for S > 0, else 0
b(t) = P_c(t) beta(t)
```

Historical `tanh` saturation, `tau / A_fact` CRM compression, signal-derived physical fallbacks, and epsilon-modified canonical denominators are not used by the reference path. An exactly constant sampled observable is still treated as the canonical null/rest-state postulate and bypasses derivative/CRM evaluation exactly.

## Installation

Core package:

```bash
pip install agencitylab
```

From a source checkout for development:

```bash
python -m pip install -e ".[dev]"
```

Optional feature groups include `viz`, `app`, `ml`, `export`, and `docs`.

## Minimal canonical example

```python
import numpy as np
from agencitylab import compute_agencity

xi = np.arange(8.0)
u = np.sin(xi)

result = compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    P_c=5.0,
    unit="rad",
    coordinate_unit="s",
    power_unit="W",
)

print(result.b.shape)
print(result.b_unit)  # W·nat
print(result.summary())
```

`A_ref`, `tau`, and `P_c` are physical/contextual quantities. Supply them explicitly, carry scalar values in metadata, or use a deliberately registered physical convention. The canonical pipeline does not infer them from signal standard deviation, MAD, range, z-score, or arbitrary defaults. The canonical CRM window is `w = tau`.

The stable API also accepts an externally specified sampled or callable `P_c(t)` when a time-dependent characteristic-power context is required. Such a profile must match the computation coordinate and is never derived from `u`.

Unit arguments are descriptive labels only: AgencityLab does not silently convert magnitudes between unit systems. `unit` applies to `u` and `A_ref`, `coordinate_unit` to `xi` and `tau`, and `power_unit` to `P_c`. The observable `b` is labelled with the corresponding informational-power unit, e.g. `W·nat` when `P_c` is in watts.

## Stable public API

`compute_agencity()` accepts one finite one-dimensional scalar observable. `data=` remains a compatibility alias for `u=`, but passing both is an explicit error. Unknown keywords are rejected rather than silently ignored.

`AgencityResult` validates its numerical payload, supports scalar or sampled `P_c`, keeps canonical wrapped `theta = angle(U)` rather than silently unwrapping phase, uses the stable `0.3` serialization schema, and exposes unit metadata. `ExperimentMetadata` preserves unknown fields for forward compatibility and keeps physical/contextual parameters separate from signal-derived quantities.

Batch items can carry per-item physical parameters and metadata. Streaming maintains monotonically increasing implicit coordinates across chunks and raises an explicit `StreamNotReadyError` when there is not yet enough CRM history.

See [`docs/stable_api.md`](docs/stable_api.md) for the computational API contract.

## Scientific validation in 0.4

The v0.4 reference bench covers seven fixed systems: exact rest, sinusoid, underdamped oscillator, Van der Pol oscillator, negative-damping unstable oscillator, low-pass-filtered Ornstein-Uhlenbeck process, and the classical Lorenz system. The suite checks stated theory consequences without tuning the canonical equations to the resulting numbers.

It also tests translation and sign-inversion invariance, temporal covariance when time and `tau` are scaled together, small- and large-amplitude limits, exact `P_c` linearity, uniform-grid convergence, and robustness to decreasing smooth perturbations.

A green scientific-validation suite means that the implementation reproduces the tested mathematical/numerical consequences under the documented benchmark conditions. It is not empirical confirmation of the Theory of Agencity itself. See [`docs/scientific_validation.md`](docs/scientific_validation.md).

## Agencity Analysis in 0.5

The v0.5 analysis layer transforms the canonical result into scientific diagnostics without changing the canonical arrays. It implements:

- theoretical local angular variance `Sigma_Theta(t) = Var(Theta(s); s in [t-tau,t])` on complete structurally valid windows;
- structural phase coherence based on canonical `Theta`, not on `arg(beta)` or `arg(b)`;
- the real-agencity criterion `S > 0`, low `Sigma_Theta`, significant `|b|`, with no universal numerical thresholds;
- signed algebraic curvature of the **beta trajectory**, with no epsilon in the curvature denominator;
- structural winding number diagnostics and explicit undefined handling across `S = 0`;
- exact agencity zeros from `S = 0` or `J = 0`, and critical-surface crossings `D = S`;
- explicit-threshold orientation jumps, local peaks of `D`, and explicit-threshold plateaus of `S`;
- threshold-free finite-record regime signatures;
- contextual classification into null, passive damped, active oscillating, unstable, stochastic, or chaotic regimes through explicit `RegimeCriteria`;
- multiscale signature fits without epsilon substitution or hidden slope thresholds.

Conservative defaults are intentional. For example:

```python
from agencitylab import analyze_agencity

analysis = analyze_agencity(result)
print(analysis["regime"])                 # undetermined for non-null data by default
print(analysis["real_agencity"]["status"])  # undetermined without context thresholds
```

When an experiment supplies physically or statistically justified thresholds, pass them explicitly and retain them with the analysis output. A single intermittent sample in noise or chaos is not enough to make a whole-record real-agencity claim unless the caller also defines an explicit persistence/fraction rule.

See [`docs/agencity_analysis.md`](docs/agencity_analysis.md) for the complete v0.5 interpretation contract.

## Multiscale & Extensions in 0.6

The v0.6 extension layer generalises the usable framework without silently modifying the scalar reference equations:

- `compute_agencity_spectrum()` returns the time-resolved spectrum `b(t, tau)` plus scalar summaries. By default every scale uses `w = tau`.
- Supplying `windows=` makes the CRM width `w` explicit and independent from `tau`, as an advanced-theory extension. The stable `compute_agencity()` path continues to reject `w != tau`.
- `optimize_agencity_window()` implements the advanced angular-stability criterion `Phi2`, searching discrete windows `w = N delta` without treating undefined structural orientation as artificial zero variance.
- `compute_discrete_agencity()` is a convenience entry point for uniformly sampled sequences and delegates to the existing canonical sampled-data engine rather than duplicating equations.
- `compute_multivariate_agencity()` computes scalar Agencity component by component, then forms the theory-specified pointwise `P_c`-weighted state and additive total flux.
- `riemannian_extension_status()` reports the Riemannian construction as **experimental and not implemented** because the source gives the geometric direction but explicitly defers the detailed analysis needed for a production pipeline.

`tau`, `w`, the sampling interval `delta`, and a multiscale scan are therefore represented as different objects. A peak in a user-supplied scale spectrum is a diagnostic result; it is not silently promoted to the physical characteristic time.

See [`docs/multiscale_extensions.md`](docs/multiscale_extensions.md) for the complete v0.6 contract and scientific boundary.

## Repository map

- `agencitylab/core/`: deterministic canonical mathematical operators plus explicitly labelled mathematical extensions. No regime interpretation belongs here.
- `agencitylab/api/`: stable user-facing orchestration; `compute_agencity` is the canonical scalar reference entry point and `extensions.py` exposes explicit v0.6 generalisations.
- `agencitylab/analysis/`: derived indicators, diagnostics, geometry, coherence, events, transitions, signatures, classifications, and reports.
- `agencitylab/models/`: reproducibility-oriented result and metadata containers.
- `tests/`: analytical, API, scientific-validation, analysis, extension, integration, regression, and foundation tests.
- `docs/`: project overview, theory mapping, stable API, scientific-validation, analysis, and extension documentation.
- `benchmarks/scientific/`: deterministic theory-facing reference systems; other benchmark folders remain experimental/performance-oriented.

## Documentation

Start with:

- [`docs/multiscale_extensions.md`](docs/multiscale_extensions.md) for the v0.6 multiscale, window, discrete, multivariate, and Riemannian boundaries.
- [`docs/agencity_analysis.md`](docs/agencity_analysis.md) for the v0.5 analysis layer and diagnostic boundaries.
- [`docs/scientific_validation.md`](docs/scientific_validation.md) for the v0.4 validation scope and reference systems.
- [`docs/stable_api.md`](docs/stable_api.md) for the public computational contract established in v0.3.
- [`docs/overview.md`](docs/overview.md) for architecture and separation between canonical physics and software layers.
- [`docs/theory_mapping.md`](docs/theory_mapping.md) for canonical definitions, parameter policy, null-state convention, and numerical boundaries.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and scientific-change rules.

## Development checks

The repository supports Python 3.10 and 3.11. CI verifies both versions with:

```bash
python -c "import agencitylab; print(agencitylab.__version__)"
ruff check agencitylab tests
pytest
python -m build
```

The Ruff policy remains a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`) so scientific changes are not obscured by unrelated style churn.

## Scientific caution

Agencity is an emerging theoretical framework. Implementation fidelity, API stability, deterministic reference validation, diagnostic tooling, and mathematically specified extensions are not empirical validation. High dynamic intensity is not, by itself, evidence of agency, and `beta != 0` does not establish coherent or "real" agencity.

Analysis thresholds, peak filters, plateau tolerances, persistence fractions, regime criteria, scale selections, and signal-derived window optima are diagnostics or explicit extensions. They must never silently become canonical physical constants.

Experimental, heuristic, diagnostic, or legacy components must remain labelled as such. The current theory documents define the canonical physics; Git history only documents previous implementations.

## Author and upstream

Theory and original project: **Gilbert BEMWIZ**.

Canonical upstream repository: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
