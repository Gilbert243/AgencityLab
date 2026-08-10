# AgencityLab

AgencityLab is an alpha-stage scientific Python project for implementing and experimenting with the theory of **Agencity** developed by Gilbert BEMWIZ.

The project is research software. Its purpose is to make the theory inspectable, testable, reproducible, and progressively comparable with data. It is **not** currently evidence that Agencity is a universally validated physical observable, a classifier of agency, or a replacement for established statistical measures.

## Status of the implementation

The current `0.1.x` line establishes the software foundations: package metadata, test discovery, CI, public imports, documentation structure, and theory-to-code traceability.

The current theory target uses the conceptual chain:

```text
u -> u* -> X* -> A* -> M, O -> D, S -> J, U -> beta -> b
```

where the current theory sources define reduced derivatives, causal memory/organisation terms, dynamic and structural intensities, a logarithmic contrast, a complex orientation, structured Agencity `beta`, and the observable `b`.

**Important:** parts of the existing numerical implementation predate the current theoretical formulation. In particular, the operands used by `M` and `O`, the use of `tanh`, and CRM window compression are not yet fully reconciled with the current theory sources. These differences are documented in [`docs/theory_mapping.md`](docs/theory_mapping.md) and are intentionally deferred to the scientific reconciliation phase rather than being silently changed in a foundations release.

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

## Minimal example

```python
import numpy as np
from agencitylab import compute_agencity

xi = np.linspace(0.0, 10.0, 200)
u = np.sin(xi)

result = compute_agencity(u=u, xi=xi)
print(result.b.shape)
```

Use explicit keyword arguments for scientific inputs. Structural parameters such as `tau`, characteristic power, normalisation choices, and physical metadata should be supplied deliberately when they are known rather than inferred from a toy example.

## Repository map

- `agencitylab/core/`: numerical operators and current computational pipeline primitives.
- `agencitylab/api/`: user-facing compute, analysis, batch, streaming, reporting, and export interfaces.
- `agencitylab/analysis/`: diagnostics and higher-level interpretation. These modules must not silently redefine canonical equations.
- `agencitylab/models/`: structured result and metadata objects.
- `tests/`: unit, integration, regression, and foundation checks.
- `docs/`: project overview, theory mapping, tutorials, API documentation, and references.
- `examples/` and `benchmarks/`: experimental material; coverage is still incomplete in the alpha series.

## Documentation

Start with:

- [`docs/overview.md`](docs/overview.md) for the software architecture and project guarantees.
- [`docs/theory_mapping.md`](docs/theory_mapping.md) for the current theory-to-code correspondence and known discrepancies.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) for development and scientific-change rules.

## Development checks

The repository currently supports Python 3.10 and 3.11. CI verifies both versions with:

```bash
python -c "import agencitylab; print(agencitylab.__version__)"
ruff check agencitylab tests
pytest
python -m build
```

The v0.1 Ruff policy is a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`) applied across the package and tests. The existing repository has broader style debt; that cleanup is intentionally separated from scientific reconciliation so linting does not trigger large unrelated rewrites.

## Scientific caution

Agencity is an emerging theoretical framework. Validation across domains remains a research programme. High dynamic intensity is not, by itself, evidence of agency; numerical outputs must be interpreted together with structural coherence, assumptions, scales, and uncertainty.

Experimental, heuristic, diagnostic, or legacy components should be labelled as such. When code and theory conflict, the conflict should be documented and resolved explicitly with reference to the theory sources and tests rather than by silently adapting the theory to existing code.

## Author and upstream

Theory and original project: **Gilbert BEMWIZ**.

Canonical upstream repository: `Gilbert243/AgencityLab`.

## License

MIT. See [`LICENSE`](LICENSE).
