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

AgencityLab `1.0.0` is the first Stable Scientific Release. The stable software contract combines the canonical scalar pipeline, deterministic scientific-validation battery, separate diagnostic analysis layer, multiscale/discrete/multivariate constructions, researcher workflows, packaging checks, executable examples, and reproducibility metadata.

Version 1.0 freezes the documented public API under Semantic Versioning. It does not redefine `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`, and software stability is not empirical confirmation of the theory.

Read [the stable API contract](stable_api.md), [the theory mapping](theory_mapping.md), [scientific validation](scientific_validation.md), and [v1.0 release readiness](release_readiness.md) before treating an implementation detail, diagnostic threshold, benchmark observation, or experimental extension as a scientific claim.

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
