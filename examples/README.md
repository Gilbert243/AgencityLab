# AgencityLab examples

## Stable v1.0 examples

The following examples are part of the v1.0 supported user workflow and are executed by CI:

- `basic/simple_compute.py` - minimal canonical compute plus diagnostics;
- `basic/explore_signal.py` - deterministic multi-signal diagnostic exploration and non-interactive visualization;
- `basic/scientific_workflow.py` - compute -> diagnostics -> figures -> exports;
- `advanced/agencity_spectrum.py` - public multiscale computation.

Install the capabilities required by the example before running it. The CI example environment uses:

```bash
pip install "agencitylab[viz,export]"
```

The built-distribution release gate separately tests public JSON/CSV export from the wheel itself.

## Research and historical scripts

Other files under `examples/advanced/`, `datasets/`, `notebooks/`, `publication/`, or `scripts/` are retained as research, historical, or development material unless they are explicitly promoted in the stable documentation and release CI. Their presence does not make them part of the v1.0 stable API contract.

Stable examples must use documented public entry points and explicit physical/contextual parameters. Research scripts must not be treated as alternate definitions of the canonical theory.
