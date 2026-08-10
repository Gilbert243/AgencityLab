# AgencityLab overview

AgencityLab has two goals that must remain distinct:

1. provide a faithful, inspectable numerical reference implementation of the selected Agencity theory definitions;
2. provide an experimental laboratory for diagnostics, alternative formulations, multiscale studies, and future validation.

The first goal requires strict traceability. The second allows exploration, but exploratory choices must not be presented as canonical without an explicit theoretical basis.

## Software layers

### `agencitylab.core`

Low-level numerical operators used by the computation pipeline: normalisation, derivatives, causal moving correlation, memory/organisation terms, dynamic and structural intensities, contrast/orientation, beta, characteristic power, and final Agencity flow.

This layer is where mathematical definitions should be reconciled with the theory sources. A module name or docstring does not, by itself, prove that an implementation is canonical.

### `agencitylab.api`

User-facing entry points such as `compute_agencity`, analysis helpers, pipeline builders, batch/streaming interfaces, reports, exports, and visualisation.

`compute_agencity` is the primary reference entry point, but in the `0.1.x` series its full scientific fidelity is still under review. See `theory_mapping.md`.

### `agencitylab.analysis`

Higher-level metrics, coherence diagnostics, regimes, information measures, events, geometry, signatures, anomalies, and multiscale tools. These modules interpret outputs; they must not silently redefine the canonical core.

Many analysis functions are exploratory. Their existence in the package should not be read as experimental validation of the underlying scientific hypotheses.

### `agencitylab.models`

Structured result and metadata containers used to make inputs, assumptions, and outputs easier to inspect and reproduce.

### `tests`

The test suite is discovered from the repository-level `tests/` directory. Foundation tests cover imports, metadata/version consistency, and public API availability. Scientific tests should be added as theory reconciliation proceeds.

## Foundation guarantees for 0.1.12

Version `0.1.12` is a software-foundation release. It is intended to guarantee that:

- package metadata and runtime version agree;
- Python 3.10 and 3.11 are the declared and CI-tested interpreter versions;
- pytest discovers the actual test suite;
- the package can be imported after installation;
- the public compute API remains importable;
- Ruff runs a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`) across the Python package and tests;
- source and wheel distributions can be built;
- GitHub Actions verifies the supported Python matrix;
- the README and contribution guide do not hide known theory/code mismatches;
- theory-to-code differences are explicitly documented before scientific changes are made.

These guarantees do **not** assert that every numerical operator already matches the latest theory formulation, that the broader legacy codebase is fully style-clean, or that Agencity has been validated across real-world domains.

## Canonical versus experimental status

Use the following labels in code review and documentation:

- **canonical**: directly implements the selected theory definition without an extra modelling assumption;
- **experimental**: deliberate alternative or extension being investigated;
- **heuristic**: practical rule introduced for robustness, inference, or convenience;
- **diagnostic**: derived interpretation used to inspect results rather than define Agencity itself;
- **legacy**: behaviour inherited from an earlier formulation and retained temporarily for compatibility or comparison.

When a component combines these categories, document the boundary explicitly.

## Next scientific phase

The next phase should focus on reconciliation rather than expansion. The highest-priority questions are the definitions of `M` and `O`, the role of `tanh`, the CRM window, and the distinction between `tau`, sampling resolution, window scale, and multiscale analysis. Those decisions should be backed by focused tests and an updated mapping rather than by a broad refactor.
