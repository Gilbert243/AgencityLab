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

AgencityLab is alpha-stage research software for implementing and experimentally studying the Theory of Agencity. Version `0.8.0` includes the canonical scalar pipeline, a stable computational API, deterministic scientific-validation systems, a separate diagnostic analysis layer, multiscale/discrete/multivariate constructions, researcher-facing workflows, and a measured engineering/performance layer.

Version 0.8 optimizes algorithms, dependency boundaries, packaging, and verification. It does not redefine `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`, and it is not empirical confirmation of the theory.

Read [the overview](overview.md), [the theory mapping](theory_mapping.md), and [the engineering report](engineering_performance.md) before treating an implementation detail, diagnostic threshold, or benchmark observation as a scientific claim.

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
