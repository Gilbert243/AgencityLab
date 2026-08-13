# AgencityLab

AgencityLab is an open-source Python framework implementing and testing the
**Theory of Agencity**.

**Current software version: 1.0.0.** This is the first stable public software
contract. Software stability does not imply empirical validation of
experimental, research, or speculative scientific layers.

## Canonical observable

The reference observable is

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

For `S > 0`, `U = (M + i O) / S`, `beta = J * U`, and `b = P_c * beta`.
For `S = 0`, the canonical convention is `U = 0` and `beta = 0`. No epsilon is
inserted into these valid equations.

## Installation

The canonical engine and stable public API require only NumPy:

```bash
pip install agencitylab
```

Optional capabilities are isolated in extras such as `scientific`, `data`,
`viz`, `export`, `numba`, and `jax`.

AgencityLab 1.0 supports CPython 3.10 through 3.14 and ships PEP 561 typing
metadata (`py.typed`).

## Quickstart

```python
import numpy as np
import agencitylab as al

xi = np.linspace(0.0, 20.0, 801)
u = np.sin(xi)

result = al.compute_agencity(
    u,
    xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
)

print(result.b)
```

`compute_agencity()` is the sole reference canonical end-to-end scalar pipeline.
`A_ref`, `tau`, `w`, and `P_c` are physical/contextual inputs; they are not
hidden in global runtime configuration. If `w` is omitted, the implementation
fallback `w = tau` is recorded explicitly and is not presented as a universal
theoretical identity.

The canonical 1.0 pipeline is NumPy based. Experimental Numba/JAX primitives
live under `agencitylab.backends`; requesting an optional acceleration library
does not silently change canonical equations or result semantics.

## Public API map

AgencityLab uses a small package root and explicit scientific namespaces:

```python
import agencitylab as al

result = al.compute_agencity(...)
analysis = al.analysis.analyze_agencity(result)
field = al.fields.compute_agencity_field(...)
```

| Task | Recommended namespace | Status |
| --- | --- | --- |
| Canonical scalar computation | `agencitylab.compute_agencity` | canonical |
| Diagnostics and interpretation | `agencitylab.analysis` | diagnostic |
| Stable workflows, batch, streaming, exports | `agencitylab.api` | software API |
| Observable spatial field | `agencitylab.fields.compute_agencity_field` | experimental |
| Autonomous/classical field physics | `agencitylab.fields` | research |
| Thermodynamics | `agencitylab.thermodynamics` | research |
| Classical gravity | `agencitylab.gravity` | research |
| Quantum field primitives | `agencitylab.quantum` | speculative |
| Homogeneous cosmology | `agencitylab.applications.cosmology` | speculative |

See [`docs/api_map.md`](docs/api_map.md) for the complete navigation map.

## Results and workflows

`AgencityResult` contains the canonical computation result and reproducibility
metadata. Diagnostic analyses, multiscale products, signatures, reports and
figures are separate workflow artifacts rather than mutable fields on the
canonical result object.

Result serialization uses schema `1.0`. Optional pandas and xarray adapters are
available through `result.to_dataframe()` and `result.to_xarray()` when the
`data` extra is installed.

## Observable and autonomous fields

`agencitylab.fields.compute_agencity_field()` applies the canonical temporal
pipeline independently at each spatial location and returns observable
`beta_obs(x,t)` and `b_obs(x,t)`. It remains **experimental**.

Promotion to the autonomous research field remains explicit:

```text
phi = sqrt(P_c * tau) * beta_obs
```

Observable `beta_obs` and autonomous `phi` are distinct objects and are not
renamed or silently merged.

The classical field equations, quartic potential, coherent structures, and
field topology remain **research**. Gravity retains its Chapter-19
`(-,+,+,+)` convention while flat Chapter-16 field dynamics retain
`(+,-,-,-)`; these signatures are not silently unified.

## Scientific status

- **canonical** — scalar Theory pipeline and its accepted identities;
- **diagnostic** — interpretation layered on canonical outputs;
- **experimental** — orchestration/numerical extensions not promoted to canonical theory;
- **research** — autonomous field, coherent, thermodynamic and gravity models;
- **speculative** — quantum/agenton and cosmological extensions.

Diagnostics consume canonical results and do not redefine them. In particular,
`beta != 0` is not by itself a criterion for coherent or “real” agencity.

## Development

```bash
python -m pip install -e ".[dev,docs]"
ruff check agencitylab tests benchmarks/performance examples
mypy --follow-imports=skip agencitylab/api/compute.py agencitylab/models/result.py
python -m pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m build
```

CI verifies Python 3.10–3.14, the declared minimum NumPy core contract, typing,
coverage measurement, wheel/sdist installation, optional extras, documentation,
examples and numerical-equivalence benchmarks.

See `CONTRIBUTING.md`, `SUPPORT.md`, and `RELEASING.md` for contribution, support,
and release policies.

## Citation and license

Scientific users should cite `CITATION.cff`. AgencityLab is distributed under
the MIT License.
