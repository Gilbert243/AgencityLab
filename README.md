# AgencityLab

AgencityLab is an alpha-stage scientific Python project for implementing and experimenting with the Theory of Agencity developed by Gilbert BEMWIZ.

The project is research software. Its purpose is to make the theory inspectable, testable, reproducible, and progressively comparable with data. It is not currently evidence that Agencity is a universally validated physical observable, a classifier of agency, or a replacement for established statistical measures.

## Status of the implementation

Version `0.4.0` keeps the v0.2 canonical scalar-signal `u -> b` core and the v0.3 stable computational API, then adds a deterministic scientific-validation battery for reference regimes, limiting cases, invariances, discrete convergence, perturbation robustness, amplitude scaling, and temporal scaling.

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

The public exception hierarchy lets applications distinguish input, physical-context, unit, batch, and streaming failures while validation errors remain compatible with existing `ValueError` handling.

Batch items can carry per-item physical parameters and metadata. Streaming maintains monotonically increasing implicit coordinates across chunks and raises an explicit `StreamNotReadyError` when there is not yet enough CRM history.

See [`docs/stable_api.md`](docs/stable_api.md) for the computational API contract.

## Scientific validation in 0.4

The v0.4 reference bench covers seven fixed systems: exact rest, sinusoid, underdamped oscillator, Van der Pol oscillator, negative-damping unstable oscillator, low-pass-filtered Ornstein-Uhlenbeck process, and the classical Lorenz system. The suite checks stated theory consequences without tuning the canonical equations to the resulting numbers.

It also tests translation and sign-inversion invariance, temporal covariance when time and `tau` are scaled together, small- and large-amplitude limits, exact `P_c` linearity, uniform-grid convergence, and robustness to decreasing smooth perturbations.

Finite-record CRM metrics exclude the initial interval before two full causal windows exist. This is a numerical boundary convention only. Stochastic tests do not assume `noise => D = 0` or `noise => beta = 0`.

Run the scientific subset with:

```bash
pytest tests/scientific
python -m benchmarks.scientific.reference_bench
```

A green scientific-validation suite means that the **implementation reproduces the tested mathematical/numerical consequences under the documented benchmark conditions**. It is not empirical confirmation of the Theory of Agencity itself. See [`docs/scientific_validation.md`](docs/scientific_validation.md).

## Repository map

- `agencitylab/core/`: deterministic canonical mathematical operators plus explicitly labelled helpers.
- `agencitylab/api/`: stable user-facing orchestration; `compute_agencity` is the canonical reference entry point.
- `agencitylab/analysis/`: diagnostics and interpretation. These modules must not silently redefine canonical equations.
- `agencitylab/models/`: reproducibility-oriented result and metadata containers.
- `tests/`: analytical, API, scientific-validation, integration, regression, and foundation tests.
- `docs/`: project overview, theory mapping, API and scientific-validation documentation, tutorials, and references.
- `benchmarks/scientific/`: deterministic theory-facing reference systems; other benchmark folders remain experimental/performance-oriented.

## Documentation

Start with:

- [`docs/scientific_validation.md`](docs/scientific_validation.md) for the v0.4 validation scope, reference systems, and scientific interpretation.
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

Agencity is an emerging theoretical framework. Implementation fidelity, API stability, and deterministic reference validation are not empirical validation. High dynamic intensity is not, by itself, evidence of agency, and `beta != 0` does not establish coherent or "real" agencity. Interpretation should use a separate diagnostic layer involving structure, orientation stability, significant `|b|`, assumptions, scales, and uncertainty.

Experimental, heuristic, diagnostic, or legacy components must remain labelled as such. The current theory documents define the canonical physics; Git history only documents previous implementations.

## Author and upstream

Theory and original project: **Gilbert BEMWIZ**.

Canonical upstream repository: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
