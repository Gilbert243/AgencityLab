# Scientific validation — v0.4

Version 0.4 adds a deterministic validation battery that checks whether the current discrete implementation reproduces consequences stated by the accepted Theory of Agencity. It does **not** claim that the theory itself has been experimentally validated.

## Scientific boundary

The validation code is downstream of the canonical core. It must never change `beta`, `J`, CRM, `M`, `O`, `D`, `S`, `A_ref`, `tau`, `w`, or `P_c` merely to make a benchmark pass. If a prediction and the implementation disagree, the first divergent stage of

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,Theta -> beta -> b
```

must be identified before any implementation change is considered.

The governing current scalar equations remain those documented in `docs/theory_mapping.md`, including `e = exp(1)` in

```text
J = ln((e + D) / (e + S)).
```

Some numerical examples in the May 2026 advanced volume were written with an earlier effective offset convention and therefore quote values such as `-ln(2)` for a damped asymptote. Those numerical constants are not used as v0.4 acceptance targets. The August 2026 second-edition canonical equation and the maintained project definition take precedence.

## Reference battery

`benchmarks/scientific/reference_bench.py` defines seven deterministic signals. Parameters are fixed by the benchmark and are never estimated from the resulting Agencity output.

| Case | Generator | Fixed context | Theory-facing expectation |
| --- | --- | --- | --- |
| Rest | `u = constant` | `A_ref=1`, `tau=1`, `P_c=1` | exact `b = 0` |
| Sinusoid | `sin(t)` | `tau=2*pi` | periodic structure and stable orientation |
| Damped oscillator | `exp(-0.1 t) sin(omega t)` | `omega0=1`, `tau=2*pi/omega` | dynamics decay; residual structure dominates (`J < 0`) |
| Van der Pol | `mu=1` | fixed ODE, burn-in, `tau=2*pi` | bounded self-sustained periodic regime |
| Unstable oscillator | negative damping `alpha=0.1` | `omega0=1` | logarithmic contrast grows asymptotically linearly, slope near `2 alpha` |
| Filtered OU | seeded OU + fixed Gaussian low-pass | `theta=2`, `sigma=0.35`, `tau=1` | non-zero local dynamics, irregular structural orientation |
| Lorenz | classical `10, 28, 8/3`, `u=x` | fixed initial condition and burn-in | bounded finite-window response with irregular orientation |

The fixed `A_ref` values are benchmark context. In particular, the Lorenz benchmark uses `A_ref=20` as an explicit model convention; it is not computed from the sampled standard deviation, range, MAD, or any other signal statistic.

## Finite-record CRM boundary

The theory defines CRM using two causal windows of history. A finite array has no samples before its first observation. The implementation therefore initializes CRM outputs to zero until two complete windows are available.

Regime metrics use the conservative mask

```text
t >= t0 + 2*tau
```

so the unknown prehistory does not contaminate scientific comparisons. This mask is a **numerical boundary convention**, not a physical rule and not a modification of CRM.

## Properties tested

The automated scientific suite checks distinct kinds of statements separately:

- **Canonical identities and exact limits:** exact rest; linear scaling with `P_c`.
- **Mathematical consequences:** translation invariance, global sign inversion invariance, temporal covariance when the physical time scale and `tau` are rescaled together, non-invariance under amplitude scaling, non-zero small-amplitude structured limit, logarithmic large-amplitude growth.
- **Reference-system behaviour:** periodic sinusoid, structure-dominated damped tail, bounded Van der Pol cycle, growing unstable response, irregular filtered stochastic and Lorenz orientations.
- **Numerical approximation:** convergence under uniform grid refinement and the first-order-or-better trend required by the discrete convergence theorem.
- **Robustness experiment:** decreasing output error for decreasing smooth perturbations around a regular periodic signal.

Numerical tolerances in these tests are acceptance tolerances for the **fixed benchmark**, not universal physical thresholds and not a definition of "real agencity".

## Discrete convergence

The current theory gives a first-order global convergence statement for the discrete construction when the observable is sufficiently smooth, the CRM window satisfies `w = N delta`, and the comparison stays away from singular regimes. The derivative stencil can be locally second order, while the discrete CRM contributes an `O(delta)` Riemann-sum error; therefore the complete functional is expected to be `O(delta)` in general.

The automated test uses a smooth periodic signal, grids with 32, 64, 128, and 256 samples per period, and the finest grid as a numerical reference. It tests decreasing max-norm error and an observed refinement order greater than 0.5. This is a regression guard for the convergence trend, not a proof of the theorem.

## Noise and stochastic systems

The robustness theorem is local: on regular domains where structural intensity and CRM variances stay away from singular values, sufficiently small `C^2` perturbations produce sufficiently small changes in `b`. The v0.4 test therefore perturbs a regular sinusoid by a deterministic smooth perturbation and verifies that halving perturbation amplitude decreases the output error.

The filtered OU benchmark is a separate regime experiment. It intentionally does **not** assert `noise => D = 0` or `noise => beta = 0`. Noise may have substantial instantaneous dynamic intensity and local non-zero Agencity. Its reference signature is evaluated through persistence and orientation irregularity, not an arbitrary universal amplitude threshold.

## Reproducibility

All stochastic generation uses a fixed seed. ODE solvers use fixed equations, initial conditions, evaluation grids, and tolerances. The reference suite has no network or external-data dependency.

Run the complete validation through normal project tests:

```bash
pytest tests/scientific
```

or print the compact benchmark metrics:

```bash
python -m benchmarks.scientific.reference_bench
```

A green v0.4 validation means that the tested implementation reproduces these stated mathematical and numerical consequences within the documented benchmark conditions. It is not evidence that speculative field, gravitational, quantum, thermodynamic, or cosmological extensions are experimentally established.
