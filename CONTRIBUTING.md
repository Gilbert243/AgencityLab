# Contributing to AgencityLab

AgencityLab is research software. Contributions should improve both software quality and scientific traceability.

## Development setup

Use Python 3.10 or 3.11, matching the versions currently declared and exercised by the project.

```bash
python -m pip install -e ".[dev]"
```

Before proposing a change, run:

```bash
ruff check agencitylab tests
pytest
python -m build
```

The same checks should pass in GitHub Actions. The v0.1 Ruff configuration is deliberately a correctness-focused baseline (`E9`, `F63`, `F7`, `F82`). Broader style cleanup should be performed in focused maintenance changes rather than mixed into scientific reconciliation.

## Branch and pull-request workflow

Do not make non-trivial feature or refactor changes directly on `main`. Create a focused branch, keep the diff limited to one coherent objective, and open a pull request. Include what changed, why it changed, the tests used, and any scientific assumptions introduced or modified.

## Scientific-change rules

The theory sources are the scientific reference; existing Python code is not automatically authoritative.

When changing a mathematical operator or interpretation:

1. Identify the theoretical definition being implemented.
2. State whether the change is **canonical**, **experimental**, **heuristic**, **diagnostic**, or **legacy compatibility**.
3. Add tests that exercise the intended behaviour without turning a modelling assumption into an undocumented axiom.
4. Update `docs/theory_mapping.md` when the mapping changes.
5. If two theory sources conflict, document the conflict and resolve it deliberately. Do not silently choose whichever definition makes the current code pass.

In particular, do not conflate sampling resolution, the characteristic time `tau`, the CRM memory window `w`, and multiscale analysis. If an implementation introduces a relation such as `w = tau / A_fact`, treat it as a modelling choice unless the selected theory source explicitly defines it as canonical.

## Tests

Tests should target AgencityLab behaviour, not merely restate NumPy behaviour. Foundation tests cover imports, version consistency, and the public API. Scientific tests should distinguish identities implied by definitions from empirical hypotheses that require validation.

Avoid tests that encode obsolete theory as a permanent invariant. If a test is intentionally retained for legacy compatibility, name and document it accordingly.

## Documentation and claims

Use precise language. Do not describe unvalidated cross-domain claims as established facts. Keep hypotheses, diagnostics, and experimental extensions clearly separate from canonical equations.

## Style

Write clear Python and documentation in English unless a file has an established different convention. Keep public APIs documented and avoid avoidable placeholders, unresolved conflict markers, or unexplained TODOs in code paths presented as production-ready.
