# AgencityLab

AgencityLab is an open-source Python framework implementing and testing the **Theory of Agencity**.

**Current software version: 1.1.7.** The canonical scalar software contract is stable. Observable-field orchestration is experimental; autonomous-field, thermodynamic and gravitational layers are research; quantum and cosmological layers are speculative. Software stability is not empirical validation of any scientific claim.

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

For `S > 0`, `U = (M + i O) / S`, `beta = J * U`, and `b = P_c * beta`. For `S = 0`, the canonical convention is `U = 0` and `beta = 0`. No epsilon is inserted into these valid equations.

## Installation

The canonical engine and normal public API require only NumPy:

```bash
pip install agencitylab
```

Optional capabilities remain isolated in extras such as `scientific`, `data`, `viz`, `export`, `numba`, and `jax`. SciPy is not a mandatory dependency.

## Quickstart

```python
import numpy as np
import agencitylab as al

xi = np.linspace(0.0, 20.0, 801)
u = np.sin(xi)

result = al.compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
)

print(result.b)
```

`compute_agencity()` is the sole reference canonical end-to-end scalar pipeline. `A_ref`, `tau`, `w`, and `P_c` are physical/contextual inputs; they are not supplied by a global runtime configuration. If `w` is omitted, the implementation fallback `w=tau` is recorded explicitly and is not a universal theoretical identity.

## Scientific API map

AgencityLab uses a small top level and explicit scientific namespaces:

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
| Observable spatial field | `agencitylab.fields.compute_agencity_field` | experimental |
| Autonomous field / classical field physics | `agencitylab.fields` | research |
| Thermodynamics | `agencitylab.thermodynamics` | research |
| Classical gravity | `agencitylab.gravity` | research |
| Quantum field primitives | `agencitylab.quantum` | speculative |
| Homogeneous cosmology | `agencitylab.applications.cosmology` | speculative |

See [`docs/api_map.md`](docs/api_map.md) for the complete navigation map and compatibility policy.

## Compatibility policy for the 1.x line

Earlier 1.1.x releases exposed many specialized symbols directly from `agencitylab`. Those locations remain available as lazy compatibility aliases in 1.1.7, but explicit access emits `DeprecationWarning` and points to the owning namespace. A plain `import agencitylab` does not emit deprecation warnings.

For example, new code should use:

```python
import agencitylab.gravity as gravity
eta = gravity.minkowski_metric()
```

rather than `agencitylab.minkowski_metric()`.

Stable scalar workflows such as batch, streaming, reports, exports, pipelines, and visualization remain available under `agencitylab.api`; this release changes navigation, not their scientific semantics.

## Observable and autonomous fields

`agencitylab.fields.compute_agencity_field()` applies the canonical temporal pipeline independently at each spatial location and returns observable `beta_obs(x,t)` and `b_obs(x,t)`. It remains **experimental**.

Promotion to the autonomous research field remains explicit:

```text
phi = sqrt(P_c * tau) * beta_obs
```

Observable `beta_obs` and autonomous `phi` are distinct objects and are not renamed or silently merged.

The classical field equations, quartic potential, coherent structures, and field topology remain **research**. Gravity retains its Chapter-19 `(-,+,+,+)` convention while flat Chapter-16 field dynamics retain `(+,-,-,-)`; this release does not reconcile those signatures.

## Runtime configuration is not physics

`agencitylab.config` now contains software/runtime options only. Historical configuration keys that looked like global physical defaults are accepted temporarily only as deprecated metadata and cannot modify the canonical calculation. Physical quantities such as `A_ref`, `tau`, `w`, `P_c`, `Gamma`, `lambda`, `mu`, temperature parameters, `xi`, and `G` must be supplied explicitly to the scientific API that owns them.

Historical YAML model/config files that were not consumed by the runtime were removed from the distributed package so they cannot be mistaken for an alternative source of theory.

## Legacy dynamics boundary

The old `agencitylab.dynamics.system` implementation was not canonical. **The theory states `beta = J * U` and `b = P_c * beta`, while the legacy implementation used tanh-based beta factors and a discrete beta variation.** Those legacy equations are retired rather than preserved for compatibility. Generic dynamical-systems utilities remain where they have independent numerical value, and the duplicated historical RK4 location forwards to the authoritative generic RK4 primitive in `agencitylab.fields.numerics`.

## Scientific status

- **canonical** — scalar Theory pipeline and its accepted identities;
- **experimental** — orchestration/numerical extensions not promoted to canonical theory;
- **research** — autonomous field, coherent, thermodynamic and gravity models;
- **speculative** — quantum/agenton and cosmological extensions.

Diagnostics consume canonical results and do not redefine them. In particular, `beta != 0` is not by itself a criterion for coherent or “real” agencity.

## Packaging and software stability

NumPy remains the only required dependency. The historical `ml` extra remains as a compatibility alias for the narrower `numba` and `jax` extras. The PyPI classifier `Development Status :: 5 - Production/Stable` describes the stable **software contract**; it does not promote experimental, research, or speculative scientific claims.

## Documentation

Start with:

- `docs/api_map.md`
- `docs/stable_api.md`
- `docs/theory_mapping.md`
- `docs/scientific_validation.md`
- `docs/tutorials/quickstart.md`

Specialized documentation includes observable/autonomous fields, coherent structures, thermodynamics, gravity, quantum, and cosmology pages.

## Development

Supported Python versions are **3.10, 3.11, and 3.12**.

```bash
python -m pip install -e ".[dev,docs]"
ruff check agencitylab tests benchmarks/performance examples
pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m build
```

See `CONTRIBUTING.md` for scientific-change rules and the GitHub workflow.

## Citation and license

Scientific users should cite `CITATION.cff`. AgencityLab is distributed under the MIT License.
