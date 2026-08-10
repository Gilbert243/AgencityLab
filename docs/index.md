# AgencityLab Documentation

```{toctree}
:maxdepth: 2
:caption: Foundations

overview
theory_mapping
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

AgencityLab is alpha-stage research software for implementing and experimentally studying the theory of Agencity.

Version `0.1.12` establishes software foundations and theory-to-code traceability. It does **not** claim that every numerical operator has already been reconciled with the current theory source or that the theory has been validated across domains.

Read [the overview](overview.md) for the software architecture and [the theory mapping](theory_mapping.md) before treating any implementation detail as canonical.

## Core target quantities

The selected current theory target uses:

- observable `u` and normalized observable `u*`;
- reduced activation `X*` and reduced activity `A*`;
- memory `M` and organisation `O`;
- dynamic intensity `D` and structural intensity `S`;
- logarithmic contrast `J` and structural direction `U`;
- structured Agencity `beta` and observable flow `b`.

Known differences between this target and the current `0.1.x` numerical path are recorded explicitly in `theory_mapping.md` and are scheduled for scientific reconciliation rather than hidden by documentation.
