# AgencityLab

AgencityLab is open-source scientific Python software for implementing, inspecting, testing, and falsifying the Theory of Agencity developed by Gilbert BEMWIZ.

**Current software status: `0.9.0` Release Candidate for the candidate-v1.0 public API.**

AgencityLab is research software. A correct implementation, green reference tests, and a stable API are not empirical confirmation that Agencity is a universally validated physical observable or a replacement for established physical or statistical quantities.

## Canonical scalar pipeline

The reference NumPy engine implements:

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,U,Theta -> beta -> b
```

with the accepted relations:

```text
u* = u / A_ref
S = sqrt(M^2 + O^2)
J = ln((e + D) / (e + S))
U = (M + i O) / S       for S > 0, else 0
beta = J U               for S > 0, else 0
b(t) = P_c(t) beta(t)
```

The CRM memory width `w` and characteristic structural time `tau` are distinct parameters. Omitting `w` uses the common software convention `w=tau`; explicitly supplying a positive `w` preserves it. No epsilon is inserted into a valid canonical denominator.

## Installation

The minimal package requires only NumPy:

```bash
pip install agencitylab
```

Optional capabilities are isolated:

```bash
pip install "agencitylab[scientific]"  # SciPy reference-system utilities
pip install "agencitylab[data]"        # pandas/xarray adapters
pip install "agencitylab[viz]"         # Matplotlib figures
pip install "agencitylab[export]"      # CSV/DataFrame, Excel and PDF workflow dependencies
pip install "agencitylab[numba]"       # experimental Numba primitives
pip install "agencitylab[jax]"         # experimental JAX primitives
pip install "agencitylab[docs]"        # Sphinx documentation build
```

NumPy is the stable complete canonical backend. Numba and JAX remain experimental primitive layers and do not silently replace the reference pipeline.

## Quickstart

```python
import numpy as np
from agencitylab import analyze_agencity, compute_agencity

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

analysis = analyze_agencity(result)

print(result.b.shape)
print(result.beta[:3])
print(result.metadata.agencitylab_version)
print(analysis["regime"])
```

The result exposes the complete inspectable computational state, including `X_star`, `A_star`, `M`, `O`, `D`, `S`, `J`, `U`, `theta`, `beta`, and `b`.

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual quantities. The stable compute API does not estimate them silently from standard deviation, MAD, range, z-score, or other ordinary signal statistics. Explicit experimental/window-selection utilities are kept separate.

## Canonical computation versus diagnostics

`agencitylab/core/` contains the deterministic mathematical engine. `agencitylab/analysis/` consumes computed results to produce coherence, geometry, events, transitions, signatures, regime diagnostics, and contextual real-agencity assessments.

A non-zero `beta` is not by itself evidence of coherent or “real” agencity. Diagnostic thresholds and persistence rules remain contextual and never modify `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b`.

## Public API freeze candidate

The v0.9 candidate-v1.0 stable contract centers on:

- `compute_agencity()`;
- `AgencityResult` and `ExperimentMetadata`;
- `analyze_agencity()` and named analysis functions;
- `run_batch()` and batch analysis helpers;
- `AgencityStream` / `stream_agencity()`;
- `compute_agencity_spectrum()`;
- `compute_discrete_agencity()` and `compute_multivariate_agencity()`;
- result/study export helpers;
- visualization and scientific-workflow orchestration;
- `AgencityPipeline` / `pipeline()`.

See [`docs/stable_api.md`](docs/stable_api.md) for the exact stable, experimental, and legacy boundaries.

## Reproducibility

A newly computed `AgencityResult` preserves:

- input coordinate and observable;
- `A_ref`, `tau`, `w`, and scalar or sampled `P_c`;
- unit/domain/system metadata;
- the AgencityLab version that produced the computation;
- requested/resolved backend information and `canonical_backend="numpy"`;
- canonical intermediate arrays;
- complex `beta` and `b` without dropping their imaginary parts.

JSON serialization round-trips complex arrays. Sample-wise CSV export uses explicit real, imaginary, and magnitude columns.

## Batch, streaming and multiscale

Batch items may carry independent physical parameters and metadata; serial and parallel execution must preserve input order and results.

Full-history `AgencityStream` recomputes the retained history after updates. Under that contract, its final result is tested against one-shot computation. A finite `window_size` intentionally changes retained history; v0.9 does **not** claim an O(1)-memory online recurrence.

`compute_agencity_spectrum()` scans explicit `tau` values. A multiscale scan is not automatic estimation of the physical characteristic time. `windows=` may keep CRM widths explicit and independent.

## Scientific validation

The deterministic validation battery covers the reference cases available in the project:

- exact rest;
- sinusoid;
- underdamped oscillator;
- Van der Pol oscillator;
- negative-damping unstable oscillator;
- low-pass-filtered Ornstein–Uhlenbeck process;
- Lorenz system.

It also tests mathematical properties and numerical convergence/robustness that have been accepted by the project. Unexpected results are treated as scientific information, not as a reason to alter the theory.

See [`docs/scientific_validation.md`](docs/scientific_validation.md).

## Release Candidate gates

CI for v0.9 verifies:

- Python 3.10, 3.11, and 3.12;
- the complete pytest suite and Ruff correctness checks;
- deterministic scientific-reference tests;
- wheel and sdist builds plus clean installs and `pip check`;
- NumPy-only minimal compute/analysis;
- isolated supported extras;
- Sphinx documentation with warnings treated as errors;
- critical executable examples;
- input -> compute -> diagnostics -> export -> restore end-to-end behavior;
- retained v0.8 scientific-equivalence benchmarks.

The v1.0 readiness checklist is maintained in [`docs/release_readiness.md`](docs/release_readiness.md).

## Documentation

Start here:

- [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md) — executable signal-to-export workflow;
- [`docs/stable_api.md`](docs/stable_api.md) — candidate-v1.0 API contract;
- [`docs/theory_mapping.md`](docs/theory_mapping.md) — theory to implementation mapping;
- [`docs/agencity_analysis.md`](docs/agencity_analysis.md) — diagnostic layer;
- [`docs/multiscale_extensions.md`](docs/multiscale_extensions.md) — multiscale/discrete/multivariate boundaries;
- [`docs/engineering_performance.md`](docs/engineering_performance.md) — v0.8 engineering evidence and limits;
- [`docs/release_readiness.md`](docs/release_readiness.md) — v1.0 readiness gate;
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution and scientific-change rules.

## Development

```bash
python -m pip install -e ".[dev,docs]"
ruff check agencitylab tests benchmarks/performance examples
pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m build
python examples/basic/simple_compute.py
python examples/advanced/agencity_spectrum.py
```

The retained performance/equivalence benchmark can be run with:

```bash
python benchmarks/performance/benchmark_v08.py --quick
```

Performance observations validate implementation behavior only; they are not physical evidence for the theory.

## Scientific and experimental boundary

Experimental or speculative extensions are not promoted merely to make the Release Candidate look broader. Numba/JAX, Riemannian work, field-theory, thermodynamic, quantum, gravitational, and cosmological extensions remain outside the stable scalar reference contract unless separately accepted and validated.

## Repository lineage

Theory and original project: **Gilbert BEMWIZ**.

Active integration repository: `somafgroup/AgencityLab`.

Original repository lineage: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
