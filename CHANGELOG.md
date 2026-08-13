# Changelog

All notable user-visible changes to AgencityLab are documented here.

AgencityLab **1.0.0 is the first stable public software release**. Repository
snapshots created before 1.0 are treated as the `0.x` development line even
when temporary internal version metadata used other numbers. Those development
identifiers are not part of the stable public compatibility contract.

## 1.0.0 — 2026-08-13

### Stable software contract

- Established the first stable package-root and namespace-first public API.
- Kept `compute_agencity` as the canonical scalar entry point.
- Kept scientific status explicit across canonical, diagnostic, experimental,
  research and speculative layers.
- Marked the distribution as typed with `py.typed` and added a public-surface
  type-checking gate.
- Declared and tested CPython 3.10, 3.11, 3.12, 3.13 and 3.14 support.
- Added a minimum-dependency job for the declared NumPy core contract.
- Added measured coverage reporting without inventing an arbitrary threshold.
- Strengthened linting for the stable public surface and retained the
  repository-wide correctness lint baseline.

### API and internal hygiene

- Removed all pre-1.0 package-root compatibility aliases instead of carrying
  deprecation machinery into the first stable release.
- Removed the scientifically incorrect historical `tanh`-based Agencity
  dynamical model entirely.
- Removed the obsolete `PhysicalSystem` model and duplicate shortcut API.
- Removed duplicate RK4 compatibility routing; the authoritative generic RK4
  primitive remains in `agencitylab.fields.numerics`.
- Made runtime software configuration strict: unknown keys are errors and
  physical/scientific parameters are not accepted as hidden config defaults.
- Limited presets to diagnostic behaviour; canonical physical inputs remain
  explicit.

### Software architecture

- Split physical/context resolution out of `compute_agencity` while preserving
  the canonical numerical sequence and equations.
- Made the canonical compute API explicitly NumPy-based instead of accepting
  backend options that did not actually replace the canonical pipeline.
- Refocused `AgencityResult` on canonical result data and validation.
- Moved result serialization and pandas/xarray adapters to `agencitylab.io`.
- Kept analysis, signatures, multiscale products and reports in workflow objects
  instead of mutating them into the canonical result model.
- Introduced result schema `1.0` with strict deserialization and no implicit
  migration from development-only payloads.

### Packaging and release engineering

- Added clean wheel and source-distribution checks, including `twine check` and
  isolated install smoke tests.
- Added a secure GitHub Release -> PyPI workflow using OIDC Trusted Publishing,
  scoped to a protected `pypi` environment and without long-lived PyPI tokens.
- Added `SUPPORT.md` and `RELEASING.md` to define the Python/API support window
  and release procedure.
- Retained optional-dependency smoke tests and reproducible numerical-equivalence
  benchmarks as release gates.

### Scientific integrity

No canonical Theory of Agencity equation is changed by the 1.0 software
hardening. In particular, the canonical definitions of CRM, `M`, `O`, `D`, `S`,
`J`, `Theta`, `beta`, `P_c`, `A_ref`, `tau`, `w` and `b = P_c * beta` are not
modified. Existing field, thermodynamic, gravitational, quantum and cosmological
formula implementations retain their previous scientific status.

## Pre-1.0 development (`0.x`)

The development line established the canonical scalar pipeline, analysis and
diagnostics, reproducibility models, batch/streaming workflows, data and export
capabilities, observable spatial fields, autonomous field models, numerical
field infrastructure, coherent structures/topology, thermodynamics, classical
Agencity gravity, quantum and cosmological research/speculative extensions, and
successive engineering/architecture cleanups.

These snapshots were laboratory development releases. The stable API lifecycle
begins at 1.0.0.
