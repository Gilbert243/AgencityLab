# Benchmarks

This directory contains performance and scientific-validation benchmarks for AgencityLab.

- `scientific/` contains deterministic theory-facing reference systems used by the v0.4+ validation suite.
- `synthetic/` contains controlled exploratory signals such as sinusoids and stochastic processes.
- `realistic/` contains mixed exploratory signals.
- `performance/benchmark_v08.py` is the reproducible v0.8 engineering benchmark.

Scientific benchmarks do not redefine canonical equations. Their fixed parameters and numerical acceptance tolerances are documented in `docs/scientific_validation.md` and must not be interpreted as universal real-agencity thresholds.

## v0.8 engineering benchmark

The v0.8 suite keeps two pre-optimization algorithms inside the benchmark file as explicit engineering references:

- the direct-window `O(N*w)` Pearson CRM;
- the direct local-unwrapping `O(N*w_tau)` implementation of the theoretical `Sigma_Theta(t)` diagnostic.

They are compared with the optimized implementations for identical inputs. The complete canonical pipeline comparison fixes `u`, `A_ref`, `tau`, `w`, and `P_c`; the `Sigma_Theta` comparison additionally verifies that the finite/undefined domain is unchanged.

Run the CI-sized workload:

```bash
python benchmarks/performance/benchmark_v08.py --quick
```

Run the larger local workload and retain machine-readable output:

```bash
python benchmarks/performance/benchmark_v08.py --output benchmark-v08.json
```

The report includes:

- median wall-clock time across repeated runs;
- approximate peak traced memory;
- CRM, complete-pipeline, and `Sigma_Theta` before/after comparisons;
- maximum absolute numerical difference across canonical arrays;
- equality of the defined `Sigma_Theta` domain;
- an optimized canonical-stage profile;
- complete analysis, four-scale, three-component multivariate, serial/threaded batch, and four-chunk streaming workloads;
- batch and streaming equivalence observations;
- Python, NumPy, platform, and benchmark parameters.

Runtime values are observations, not CI thresholds. Shared runners, CPU frequency scaling, BLAS builds, and background load can change timings. CI rejects unexplained numerical divergence, but it never fails merely because a workload took longer. The retained reference algorithms exist only for measurement and equivalence checking; they are not alternate theory definitions.
