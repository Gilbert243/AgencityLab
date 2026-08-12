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

AgencityLab `1.1.1` keeps the 1.x canonical scalar software contract stable, retains **Observable Spatial Agencity Fields** as an explicitly experimental API, and integrates the common foundations for an autonomous dynamical `phi` field as a **research** layer. Software stability is distinct from empirical validation of the theory.

The research bridge from `beta_obs` to `phi` is explicit and uses `phi = sqrt(P_c * tau) * beta`; `compute_agencity_field()` never performs that promotion automatically. Version 1.1.1 also exposes the quartic reference potential, vacuum and energy primitives, research field state/solution models, parameter provenance, and generic NumPy-only field numerics. It does **not** yet provide a physical KG/TDGL PDE solver.

Version 1.0 froze the documented public scalar API under Semantic Versioning. The 1.1.x series adds backwards-compatible experimental/research functionality without redefining `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.

Read [the stable API contract](stable_api.md), [observable spatial fields](observable_fields.md), [dynamical field foundations](dynamical_field_foundations.md), [field extension contracts](field_extension_contracts.md), [the theory mapping](theory_mapping.md), [scientific validation](scientific_validation.md), and [release readiness](release_readiness.md) before treating an implementation detail, diagnostic threshold, benchmark observation, or research/speculative extension as a scientific claim.

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
