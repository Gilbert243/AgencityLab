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

stable_api
observable_fields
dynamical_field_foundations
classical_field_dynamics
coherent_structures
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

AgencityLab `1.1.4` keeps the 1.x canonical scalar software contract stable, retains **Observable Spatial Agencity Fields** as an explicitly experimental API, exposes the autonomous classical `phi` field, its reference dynamics, coherent structures, thermodynamic evaluators, and classical gravity primitives as **research** layers, and exposes limited quantum-field and homogeneous FLRW cosmology primitives as explicitly **speculative** layers. Software stability is distinct from empirical validation of the theory.

The research bridge from `beta_obs` to `phi` remains explicit and uses `phi = sqrt(P_c * tau) * beta`; `compute_agencity_field()` never performs that promotion automatically. Version 1.1.2 added conservative Klein-Gordon, dissipative Klein-Gordon, overdamped TDGL, real-sector domain-wall references, U(1) vortex references, and spatial winding diagnostics. Version 1.1.3 added source-defined thermodynamic evaluators and limited classical curved-spacetime primitives. Version 1.1.4 adds only the operational Chapter-21 quantum primitives and the homogeneous flat-FLRW Chapter-22 application without promoting either layer beyond speculative status.

The gravity package follows the Chapter-19 `(-,+,+,+)` convention explicitly. The classical flat-field dynamics preserve the Chapter-16 `(+,-,-,-)` convention. AgencityLab does not silently identify those source conventions or change either equation merely to force equality.

Version 1.0 froze the documented public scalar API under Semantic Versioning. The 1.1.x series adds backwards-compatible experimental/research/speculative functionality without redefining `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.

Read [the stable API contract](stable_api.md), [observable spatial fields](observable_fields.md), [dynamical field foundations](dynamical_field_foundations.md), [classical field dynamics](classical_field_dynamics.md), [coherent structures and topology](coherent_structures.md), [thermodynamics](thermodynamics.md), [classical gravity](gravity.md), [quantum primitives](quantum.md), [cosmological application](cosmology.md), [field extension contracts](field_extension_contracts.md), [the theory mapping](theory_mapping.md), [scientific validation](scientific_validation.md), and [release readiness](release_readiness.md) before treating an implementation detail, diagnostic threshold, benchmark observation, research model, or speculative extension as a scientific claim.

## Canonical target quantities

The reference scalar construction uses:

- observable `u` and normalized observable `u*`;
- reduced activation `X*` and reduced activity `A*`;
- memory `M` and organisation `O` from causal moving correlation;
- dynamic intensity `D` and structural intensity `S`;
- logarithmic contrast `J` and structural direction `U = exp(i Theta)` when `S > 0`;
- intrinsic state `beta`, with `beta = 0` when `S = 0`;
- observable agencity flux `b = P_c beta`.

Canonical computation, numerical safeguards, diagnostics, heuristics, experimental extensions, research models, speculative extensions, and legacy compatibility paths are labelled separately throughout the documentation.
