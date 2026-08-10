# AgencityLab

AgencityLab is an alpha-stage scientific Python project for implementing and experimenting with the Theory of Agencity developed by Gilbert BEMWIZ.

The project is research software. Its purpose is to make the theory inspectable, testable, reproducible, and progressively comparable with data. It is not currently evidence that Agencity is a universally validated physical observable, a classifier of agency, or a replacement for established statistical measures.

## Status of the implementation

Version `0.2.0` establishes the canonical scalar `u -> b` reference core against the current second-edition theory.

```text
u -> u* -> X* -> A* -> M, O -> D, S -> J, U -> beta -> b
```

The canonical path implements:

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
b = P_c beta
```

Historical `tanh` saturation, `tau / A_fact` CRM compression, signal-derived physical fallbacks, and epsilon-modified canonical denominators are not used by the v0.2 reference path. See [`docs/theory_mapping.md`](docs/theory_mapping.md) for exact theory-to-code traceability.

An exactly constant sampled observable is treated as the canonical null/rest-state postulate. The pipeline detects that state before numerical differentiation or CRM and returns `X*=A*=M=O=D=S=J=U=beta=b=0` exactly. This is an exact check, not a tolerance for near-constant signals.

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

xi = np.linspace(0.0, 10.0, 101)
u = np.sin(xi)

result = compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    P_c=1.0,
)

print(result.b.shape)
```

`A_ref`, `tau`, and `P_c` are physical/contextual quantities. Supply them explicitly, carry them in metadata, or use a deliberately registered physical convention. The canonical pipeline does not infer them from signal standard deviation, MAD, range, z-score, or arbitrary defaults. The canonical CRM window is `w = tau`.

## Repository map

- `agencitylab/core/`: deterministic canonical mathematical operators plus explicitly labelled helpers.
- `agencitylab/api/`: stable user-facing orchestration; `compute_agencity` is the canonical reference entry point.
- `agencitylab/analysis/`: diagnostics and interpretation. These modules must not silently redefine canonical equations.
- `agencitylab/models/`: reproducibility-oriented result and metadata containers.
- `tests/`: analytical unit tests, integration tests, regressions, and software-foundation checks.
- `docs/`: project overview, theory mapping, tutorials, API documentation, and references.
- `examples/` and `benchmarks/`: experimental material; coverage remains incomplete in the alpha series.

## Documentation

Start with:

- [`docs/overview.md`](docs/overview.md) for architecture and v0.2 guarantees.
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

Agencity is an emerging theoretical framework. Implementation fidelity is not empirical validation. High dynamic intensity is not, by itself, evidence of agency, and `beta != 0` does not establish coherent or "real" agencity. Interpretation should use a separate diagnostic layer involving structure, orientation stability, significant `|b|`, assumptions, scales, and uncertainty.

Experimental, heuristic, diagnostic, or legacy components must remain labelled as such. The current theory documents define the canonical physics; Git history only documents previous implementations.

## Author and upstream

Theory and original project: **Gilbert BEMWIZ**.

Canonical upstream repository: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
