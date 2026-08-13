# Contributing to AgencityLab

AgencityLab is research software. Contributions should improve both software quality and scientific traceability.

## Development setup

Use Python 3.10, 3.11, or 3.12, matching the versions officially declared and exercised by the project.

```bash
python -m pip install -e ".[dev,docs]"
```

Before proposing a change, run the same critical checks used by CI:

```bash
ruff check agencitylab tests benchmarks/performance examples
pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m build
```

The Ruff configuration is deliberately a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`). Broader style cleanup should be performed in focused maintenance changes rather than mixed into scientific reconciliation.

## Branch and pull-request workflow

Do not make non-trivial feature or refactor changes directly on `main`. Create a focused branch, keep the diff limited to one coherent objective, and open a pull request. Include what changed, why it changed, the tests used, and any scientific assumptions introduced or modified.

## Stable API and Semantic Versioning

AgencityLab 1.0 establishes the stable public API documented in `docs/stable_api.md`.

After 1.0:

- patch releases (`1.0.x`) are for bug fixes, documentation, and compatible internal improvements;
- minor releases (`1.x.0`) may add backwards-compatible functionality and public APIs;
- major releases (`2.0.0` and later) are required for intentional breaking changes to the stable public contract.

Interfaces explicitly labelled experimental, research, speculative, or legacy compatibility are not granted the same stability promise. Their status must remain visible in documentation and release notes. Do not silently promote an experimental interface into the stable contract.

New stable public APIs require documentation, tests, a reproducibility review where relevant, and a clear scientific status. Avoid unnecessary API expansion.

## Scientific-change rules

The theory sources are the scientific reference; existing Python code is not automatically authoritative.

When changing a mathematical operator or interpretation:

1. Identify the theoretical definition being implemented.
2. State whether the change is **canonical**, **experimental**, **heuristic**, **diagnostic**, or **legacy compatibility**.
3. Add tests that exercise the intended behaviour without turning a modelling assumption into an undocumented axiom.
4. Update `docs/theory_mapping.md` when the mapping changes.
5. If two theory sources conflict, document the conflict and resolve it deliberately. Do not silently choose whichever definition makes the current code pass.

In particular, do not conflate sampling resolution, the characteristic time `tau`, the CRM memory window `w`, and multiscale analysis. If an implementation introduces a relation such as `w = tau / A_fact`, treat it as a modelling choice unless the selected theory source explicitly defines it as canonical.

Physical/contextual quantities such as `A_ref`, `tau`, `w`, and `P_c` must not be silently inferred from ordinary signal statistics in the stable compute path. A numerical safeguard may prevent invalid machine operations or select a safer numerical method, but it must not redefine a valid canonical equation.

## Tests

Tests should target AgencityLab behaviour, not merely restate NumPy behaviour. Foundation tests cover imports, version consistency, and the public API. Scientific tests should distinguish identities implied by definitions from empirical hypotheses that require validation.

Changes to the canonical engine should test the earliest affected pipeline stage and relevant downstream identities. Changes to diagnostics should test that canonical arrays are not mutated or redefined. Changes to batch, streaming, multiscale, discrete, multivariate, export, or packaging behaviour should include a public-API regression test when applicable.

Avoid tests that encode obsolete theory as a permanent invariant. If a test is intentionally retained for legacy compatibility, name and document it accordingly.

## Documentation and claims

Use precise language. Do not describe unvalidated cross-domain claims as established facts. Keep hypotheses, diagnostics, and experimental extensions clearly separate from canonical equations.

Documentation for supported examples must list required optional extras. Public examples should use stable entry points instead of internal `agencitylab.core.*` imports unless the example is explicitly about implementation internals.

## Style

Write clear Python and documentation in English unless a file has an established different convention. Keep public APIs documented and avoid avoidable placeholders, unresolved conflict markers, unexplained TODOs, debug prints, or temporary files in code paths presented as production-ready.
