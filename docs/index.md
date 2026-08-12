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

AgencityLab `1.1.5` keeps the 1.x canonical scalar software contract stable. Observable spatial orchestration and generic field numerics remain explicitly **experimental**; the autonomous classical `phi` field, its dynamics, coherent structures, Chapter-15 effective-beta model, flat-field conservation laws, thermodynamic evaluators, and classical gravity primitives remain **research**; limited quantum-field and homogeneous FLRW cosmology primitives remain **speculative**. Software implementation status is distinct from empirical validation of the theory.

The research bridge from `beta_obs` to `phi` remains explicit and uses `phi = sqrt(P_c * tau) * beta`; `compute_agencity_field()` never performs that promotion automatically. Version 1.1.2 added classical autonomous-field dynamics and coherent structures, 1.1.3 added thermodynamics and classical gravity, and 1.1.4 added the operational quantum and homogeneous FLRW primitives. Version 1.1.5 adds source-defined material that remained outside those milestones: the separate Chapter-15 effective-beta equation, Chapter-16 flat-field Noether/energy-momentum primitives, the separately named Appendix-B beta-field formulation, Definition-12.4 Riemannian kinematic primitives, the Chapter-13 `Phi1` and `Phi3` criteria, and the Chapter-17 dimensionless coherent-field formulation.

The gravity package follows the Chapter-19 `(-,+,+,+)` convention explicitly. The Chapter-16 flat-field conservation package preserves `(+,-,-,-)`. AgencityLab does not silently identify those source conventions or change either equation merely to force equality.

Source-level ambiguities are documented rather than guessed. In particular, the printed Chapter-15 phase equation does not algebraically match the unambiguous complex equation without an additional factor, the Appendix-B beta normalization is kept separate from `phi = sqrt(P_c tau) beta`, and the extracted Chapter-8 cycle-area typography is not silently repaired. The source also explicitly leaves a closed autonomous equation for `b` as future work, so AgencityLab does not fabricate one.

Version 1.0 froze the documented public scalar API under Semantic Versioning. The 1.1.x series adds labelled experimental/research/speculative functionality without redefining `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.

Read [the stable API contract](stable_api.md), [observable spatial fields](observable_fields.md), [dynamical field foundations](dynamical_field_foundations.md), [classical field dynamics](classical_field_dynamics.md), [effective beta field and conservation laws](effective_beta_and_conservation.md), [coherent structures and topology](coherent_structures.md), [mathematical extensions](mathematical_extensions.md), [thermodynamics](thermodynamics.md), [classical gravity](gravity.md), [quantum primitives](quantum.md), [cosmological application](cosmology.md), [field extension contracts](field_extension_contracts.md), [the theory mapping](theory_mapping.md), [scientific validation](scientific_validation.md), and [release readiness](release_readiness.md) before treating an implementation detail, diagnostic threshold, benchmark observation, research model, or speculative extension as a scientific claim.

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
