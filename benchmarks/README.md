# Benchmarks

This directory contains performance and scientific-validation benchmarks for AgencityLab.

- `scientific/` -> deterministic theory-facing reference systems used by the v0.4 validation suite.
- `synthetic/` -> controlled exploratory signals (sinusoid, noise, etc.).
- `realistic/` -> mixed exploratory signals.
- `performance/` -> runtime and scaling tests.

Scientific benchmarks do not redefine canonical equations. Their fixed parameters and numerical acceptance tolerances are documented in `docs/scientific_validation.md` and must not be interpreted as universal real-agencity thresholds.
