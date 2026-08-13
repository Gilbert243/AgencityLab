# AgencityLab Documentation

```{toctree}
:maxdepth: 2
:caption: Foundations

overview
theory_mapping
scientific_validation
```

```{toctree}
:maxdepth: 2
:caption: Scientific computation and analysis

api_map
stable_api
observable_fields
dynamical_field_foundations
classical_field_dynamics
effective_beta_and_conservation
coherent_structures
mathematical_extensions
thermodynamics
gravity
quantum
cosmology
field_extension_contracts
agencity_analysis
multiscale_extensions
scientific_ux
engineering_performance
release_readiness
```

```{toctree}
:maxdepth: 2
:caption: Reference and tutorials

theory/agencity
theory/equations
theory/principles
theory/interpretation
tutorials/quickstart
tutorials/preprocessing
tutorials/full_pipeline
api/index
examples/index
references/bibliography
references/glossary
```

## Project status

AgencityLab `1.1.7` keeps the canonical scalar software contract stable while consolidating navigation around explicit scientific namespaces. The architecture cleanup does not modify canonical, field, thermodynamic, gravitational, quantum, or cosmological equations.

Use the [scientific API map](api_map.md) first. The main scalar entry point is `agencitylab.compute_agencity`; interpretation belongs under `agencitylab.analysis`; spatial/autonomous field work under `agencitylab.fields`; thermodynamics, gravity, quantum and cosmology under their corresponding namespaces.

Observable spatial orchestration and generic field numerics remain **experimental**. Autonomous `phi`, effective-beta dynamics, flat-field conservation, coherent structures, thermodynamics and classical gravity remain **research**. Quantum/agenton and homogeneous cosmology remain **speculative**. Software stability is distinct from empirical validation.

The bridge from observable `beta_obs` to autonomous `phi` remains explicit: `phi = sqrt(P_c * tau) * beta`. The Chapter-16 flat-field `(+,-,-,-)` convention and Chapter-19 gravity `(-,+,+,+)` convention remain explicitly distinct.

Historical physical-looking YAML configuration files are no longer distributed. Runtime configuration contains software options only, and physical/contextual quantities must be supplied explicitly to the scientific API that owns them.

The old `agencitylab.dynamics.system` model is explicitly retired: it used a tanh-based beta and a discrete beta variation that do not match the canonical definitions `beta = J*U` and `b = P_c*beta`. Generic numerical utilities remain where they are scientifically neutral.

Version 1.0 froze the documented scalar public software contract under Semantic Versioning. Specialized 1.1.x research/speculative APIs remain status-labelled, and old top-level locations are retained in 1.1.7 as deprecated compatibility aliases while new code uses the owning namespace.

## Canonical target quantities

The reference scalar construction uses observable `u`, normalized `u*`, activation `X*`, activity `A*`, memory `M`, organisation `O`, dynamic intensity `D`, structural intensity `S`, logarithmic contrast `J`, structural orientation `Theta`, intrinsic state `beta`, and flux `b = P_c beta`.

Canonical computation, numerical safeguards, diagnostics, experimental extensions, research models, speculative extensions, and legacy compatibility paths remain separately labelled.
