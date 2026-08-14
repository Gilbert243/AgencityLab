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

reference_library
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

AgencityLab `1.0.0` is the first stable public software release. Repository
snapshots that preceded 1.0 are development history and do not create stable
compatibility obligations.

Use the [scientific API map](api_map.md) first. The main scalar entry point is
`agencitylab.compute_agencity`; interpretation belongs under
`agencitylab.analysis`; spatial/autonomous field work under
`agencitylab.fields`; thermodynamics, gravity, quantum and cosmology under their
corresponding namespaces.

Observable spatial orchestration and generic field numerics remain
**experimental**. Autonomous `phi`, effective-beta dynamics, flat-field
conservation, coherent structures, thermodynamics and classical gravity remain
**research**. Quantum/agenton and homogeneous cosmology remain **speculative**.
Software stability is distinct from empirical validation.

The bridge from observable `beta_obs` to autonomous `phi` remains explicit:
`phi = sqrt(P_c * tau) * beta`. The Chapter-16 flat-field `(+,-,-,-)` convention
and Chapter-19 gravity `(-,+,+,+)` convention remain explicitly distinct.

Runtime configuration contains software options only. Physical/contextual
quantities must be supplied explicitly to the scientific API that owns them.
The stable canonical scalar pipeline is NumPy based; optional JAX and Numba
components are explicitly separate acceleration primitives rather than hidden
alternate canonical pipelines.

The incorrect pre-1.0 tanh-based Agencity dynamical model is not distributed in
1.0. Generic dynamical-system helpers remain where they are scientifically
neutral.

`AgencityResult` contains canonical result data and reproducibility metadata.
Diagnostics, signatures, multiscale products and reports remain separate
workflow artifacts. Stable result serialization begins with schema `1.0`.

## Canonical target quantities

The reference scalar construction uses observable `u`, normalized `u*`,
activation `X*`, activity `A*`, memory `M`, organisation `O`, dynamic intensity
`D`, structural intensity `S`, logarithmic contrast `J`, structural orientation
`Theta`, intrinsic state `beta`, and flux `b = P_c beta`.

Canonical computation, numerical safeguards, diagnostics, experimental
extensions, research models and speculative extensions remain separately
labelled.
