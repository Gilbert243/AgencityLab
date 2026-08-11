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

AgencityLab `1.1.0` keeps the 1.x canonical scalar software contract stable and adds **Observable Spatial Agencity Fields** as an explicitly experimental API. Software stability is distinct from empirical validation of the theory, and the spatial orchestration does not promote the future autonomous dynamical field to stable or canonical status.

Version 1.0 froze the documented public scalar API under Semantic Versioning. Version 1.1 adds backwards-compatible functionality without redefining `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.

Read [the stable API contract](stable_api.md), [observable spatial fields](observable_fields.md), [the theory mapping](theory_mapping.md), [scientific validation](scientific_validation.md), and [release readiness](release_readiness.md) before treating an implementation detail, diagnostic threshold, benchmark observation, or experimental extension as a scientific claim.

## Canonical target quantities

The reference scalar construction uses:

- observable `u` and normalized observable `u*`;
- reduced activation `X*` and reduced activity `A*`;
- memory `M` and organisation `O` from causal moving correlation;
- dynamic intensity `D` and structural intensity `S`;
- logarithmic contrast `J` and structural direction `U = exp(i Theta)` when `S > 0`;
- intrinsic state `beta`, with `beta = 0` when `S = 0`;
- observable agencity flux `b = P_c beta`.

Canonical computation, numerical safeguards, diagnostics, heuristics, experimental extensions, and legacy compatibility paths are labelled separately throughout the documentation.
