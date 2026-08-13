# Engineering & Performance — v0.8

## Scope

AgencityLab 0.8 is an implementation-engineering milestone. It improves algorithmic complexity, observability, dependency boundaries, packaging, and verification while preserving the accepted Theory of Agencity.

For identical `u`, `A_ref`, `tau`, `w`, and `P_c`, the optimized reference path must reproduce the same canonical quantities within justified floating-point precision:

```text
u* -> X* -> A* -> M,O -> D,S -> J,U,Theta -> beta -> b
```

The implementation invariant remains:

```text
S = sqrt(M^2 + O^2)
Theta = atan2(O, M)
U = (M + i O) / S  when S > 0, else 0
J = ln((e + D) / (e + S))
beta = J U          when S > 0, else 0
b = P_c beta
```

No benchmark result was used to redefine `CRM`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, `b`, `A_ref`, `tau`, `w`, or `P_c`.

## Baseline audit

The v0.7.0 baseline exposed five engineering issues.

1. **Canonical CRM complexity.** Every output sample recomputed two means, two variances, and one covariance over adjacent windows in Python. The cost was `O(N*w)`.
2. **Theoretical angular-variance complexity.** `Sigma_Theta(t)` repeated `searchsorted`, local unwrapping, and variance over every interval `[t-tau,t]`, giving `O(N*w_tau)` work.
3. **Backend inconsistency.** Optional NumPy, Numba, and JAX CRM helpers used epsilon-based variance classification; the historical Numba path also inserted epsilon into a denominator. The theory states exact zero empirical variance, while those experimental helpers implemented a threshold-modified rule.
4. **Dependency leakage.** The default analysis import eagerly required SciPy, so an advertised minimal installation could not actually import and analyze results with NumPy alone.
5. **Verification gaps.** Python 3.12, clean wheel/source installation, isolated extras, long signals, batch/streaming equivalence, and reproducible before/after benchmarks were not all covered by CI.

## Implemented optimizations

### Canonical CRM

The ordinary reference path now uses rolling first and second moments. Its work is `O(N)` in signal length for a fixed sampled record, independent of the CRM width in the ordinary path.

Scientific and numerical safeguards remain distinct:

- exactly constant sampled windows are detected by exact equality and return correlation zero;
- very short windows use the direct centred Pearson definition, remaining `O(N)` because their width is bounded;
- numerically delicate rolling-moment subtractions are recomputed with the direct centred definition;
- no epsilon is inserted into the Pearson denominator;
- the finite-record warm-up and adjacent-window geometry are unchanged.

### Theoretical `Sigma_Theta`

The optimized analysis path preserves the accepted definition:

```text
Sigma_Theta(t) = Var(Theta(s); s in [t-tau,t])
```

It vectorizes interval discovery, unwraps the angular sequence once, and evaluates interval moments with prefix sums. A global unwrap restricted to a structurally valid local interval differs from a local unwrap only by a constant integer multiple of `2*pi`; ordinary variance is invariant under that constant shift. Windows with undefined orientation remain undefined, and numerically delicate intervals fall back to the direct local definition.

This is an analysis optimization. It does not change canonical `Theta`, `beta`, or `b`.

### Optional event filtering

Unfiltered local maxima of `D` now use a NumPy-only `O(N)` detector, including the established midpoint convention for flat peaks. Explicit `prominence` or `distance` filtering retains SciPy's `find_peaks` semantics and lazily requires the `scientific` extra. The diagnostic meaning is unchanged.

## Reproducible benchmark

The CI-sized benchmark was executed on 2026-08-10 with:

- Python `3.12.13`;
- NumPy `2.5.2`;
- Ubuntu 24.04 GitHub-hosted Azure runner;
- `A_ref=1.5`, `tau=64`, `w=64`, `P_c=2.5`;
- two repetitions, reporting the median;
- deterministic synthetic input at `N = 4,096`, `16,384`, and `65,536`.

Run it with:

```bash
python benchmarks/performance/benchmark_v08.py --quick \
  --output benchmark-v08.json
```

The benchmark retains the pre-optimization algorithms locally as engineering references. They are not alternate theory definitions. CI gates numerical equivalence and equality of the defined `Sigma_Theta` domain; it does not gate wall-clock time.

### CRM before and after

| N | Direct reference | Optimized | Observed speedup | Maximum absolute difference | Traced peak memory |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.072492 s | 0.000509 s | 142.44x | `1.471e-11` | 0.10 -> 0.57 MiB |
| 16,384 | 0.291551 s | 0.001806 s | 161.41x | `7.693e-11` | 0.38 -> 2.30 MiB |
| 65,536 | 1.172867 s | 0.007051 s | 166.33x | `2.996e-10` | 1.50 -> 9.24 MiB |

### Complete canonical pipeline before and after

Only the historical CRM implementation is substituted in the reference pipeline. All equations and physical parameters are otherwise identical.

| N | Direct-CRM pipeline | Optimized pipeline | Observed speedup | Maximum absolute difference across canonical arrays | Traced peak memory |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.146294 s | 0.002565 s | 57.04x | `8.707e-11` | 0.60 -> 0.83 MiB |
| 16,384 | 0.588848 s | 0.006396 s | 92.06x | `5.706e-10` | 2.27 -> 3.33 MiB |
| 65,536 | 2.345934 s | 0.022426 s | 104.61x | `3.971e-09` | 8.70 -> 13.31 MiB |

The compared arrays are `u_star`, `X_star`, `A_star`, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, and `b`. The largest observed difference remains below the CI acceptance bound of `1e-8` on this benchmark. Exact mathematical identities such as `S=hypot(M,O)`, `beta=J*U`, and `b=P_c*beta` are tested independently.

### `Sigma_Theta` before and after

| N | Direct local definition | Optimized | Observed speedup | Maximum absolute difference | Defined domain | Traced peak memory |
|---:|---:|---:|---:|---:|:---:|---:|
| 4,096 | 0.199564 s | 0.000551 s | 362.50x | `4.452e-14` | identical | 0.05 -> 0.70 MiB |
| 16,384 | 0.825467 s | 0.001419 s | 581.59x | `1.829e-13` | identical | 0.14 -> 2.85 MiB |
| 65,536 | 3.331313 s | 0.005459 s | 610.27x | `9.038e-13` | identical | 0.52 -> 11.43 MiB |

### Optimized canonical stage profile at 65,536 samples

| Stage | Observed time |
|---|---:|
| Normalization | 0.0886 ms |
| `X*` and `A*` derivatives | 0.7117 ms |
| Memory `M` | 2.9974 ms |
| Organisation `O` | 3.7820 ms |
| `D`, `S`, `J`, `beta`, and `b` | 3.8671 ms |

This stage profile times the numerical stages directly. The complete public API timing also includes validation, metadata construction, result assembly, and orchestration.

### Representative workloads at 4,096 samples

| Workload | Observed time | Traced peak memory | Equivalence observation |
|---|---:|---:|---|
| Complete analysis | 0.063474 s | 0.752 MiB | consumes the unchanged computed result |
| Four-scale spectrum | 0.007187 s | 3.019 MiB | rows tested against independent scalar computations |
| Three-component multivariate construction | 0.005612 s | 2.296 MiB | component physics preserved |
| Four-item serial batch | 0.008245 s | 2.361 MiB | input order and per-item context preserved |
| Four-item, two-thread batch | 0.017912 s | not separately gated | maximum difference from serial: `0.0` |
| Four-chunk full-history stream | 0.007356 s | 1.429 MiB | final maximum difference from one-shot: `0.0` |

The small threaded workload is slower than serial because scheduling overhead dominates. AgencityLab does not claim universal thread acceleration.

## Memory interpretation

The optimized algorithms exchange Python-loop work for vectorized temporary arrays. On the benchmarked records, wall-clock time falls sharply while traced peak memory rises. This is an intentional and documented trade-off, not a memory optimization claim.

`tracemalloc` observes Python-traced allocations and is not total process RSS, device memory, or allocator-resident memory. Future release-candidate work should add cross-platform RSS measurements before making stronger memory claims.

## Long signals, batch, streaming, and multiscale behavior

The test suite includes:

- a 100,000-sample complete canonical computation with finite outputs and exact canonical identities;
- a 100,000-sample high-winding `Sigma_Theta` diagnostic;
- serial/threaded batch equality with independent `A_ref`, `tau`, `w`, and `P_c` per item;
- full-history streaming equality with one-shot computation;
- multiscale rows matched to independent canonical computations.

`AgencityStream` deliberately recomputes the retained record after each update. With `window_size=None`, this preserves exact full-history final equality but is not a constant-memory online recurrence. Supplying `window_size` bounds retained memory explicitly and therefore changes the retained-history problem; it is never applied silently.

## Backend contract

| Backend | Status | Scope | Complete canonical pipeline | Default precision note |
|---|---|---|:---:|---|
| NumPy | stable | reference numerical primitives and public pipeline | yes | float64 reference path |
| Numba | experimental | optional one-dimensional primitives | no | float64 primitive path |
| JAX | experimental | optional autodiff/vectorization primitives | no | normally float32 unless JAX x64 is enabled |

`backend_capabilities()` reports availability, status, scope, precision note, and ownership of the canonical pipeline. Requesting Numba or JAX validates and records an optional primitive backend; it does not silently replace the full NumPy reference pipeline.

No GPU or accelerator speed claim is made in v0.8. A future accelerated pipeline must first demonstrate equation-by-equation equivalence and explicit precision behavior.

## Dependency and installation contract

The minimal runtime dependency is NumPy:

```bash
pip install agencitylab
```

Optional capabilities are isolated:

| Extra | Purpose |
|---|---|
| `scientific` | SciPy reference systems and filtered peak diagnostics |
| `data` | pandas and xarray adapters |
| `viz` | Matplotlib figures |
| `export` | pandas, OpenPyXL, and ReportLab exports |
| `numba` | experimental Numba primitives |
| `jax` | experimental JAX primitives |
| `docs` | Sphinx documentation build |
| `dev` | complete repository test and development stack |

The historical `ml` extra remains a compatibility alias for combined Numba/JAX installation. New environments should use the narrower extras.

## CI and packaging verification

The v0.8 workflow verifies:

- import, Ruff, and the full test suite on Python 3.10, 3.11, and 3.12;
- wheel and source-distribution builds;
- clean wheel installation with SciPy, pandas, Matplotlib, Numba, and JAX absent;
- a minimal canonical computation and default analysis in that NumPy-only environment;
- clean source-distribution installation;
- isolated smoke tests for `data`, `viz`, `export`, `numba`, and `jax`;
- the reproducible benchmark and its scientific-equivalence gates;
- uploaded wheel, source-distribution, and benchmark artifacts.

Local verification commands are:

```bash
python -m pip install -e ".[dev]"
ruff check agencitylab tests benchmarks/performance
pytest
python -m build
python benchmarks/performance/benchmark_v08.py --quick
```

## Known limits before v0.9

- Vectorized CRM and `Sigma_Theta` use more transient host memory than the historical Python loops.
- Full-history streaming recomputes the retained record and is not yet a mathematically equivalent incremental recurrence.
- Numba and JAX remain experimental primitive layers rather than complete alternative pipelines.
- JAX precision and device-specific behavior require explicit study before reference use.
- The benchmark covers deterministic representative workloads on one hosted-runner class; it is not a universal hardware guarantee.
- Threaded batch performance depends on item size, worker count, NumPy behavior, and scheduling overhead.
- No distributed, GPU, or out-of-core canonical benchmark is claimed.

These limits are engineering boundaries. They do not justify changing the accepted theory.

## Scientific interpretation

Performance results validate implementation behavior only. They are not evidence that the Theory of Agencity is empirically confirmed. Non-zero `beta`, high `D`, a fast computation, or a successful benchmark does not establish coherent or real agencity. Diagnostic thresholds and persistence rules remain contextual and separate from the canonical engine.
