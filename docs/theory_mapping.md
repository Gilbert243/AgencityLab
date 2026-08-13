# Theory-to-code mapping

This document records the relationship between the accepted Agencity theory, explicit maintainer decisions, and the Python reference implementation. The primary scientific sources are *Théorie de l’Agencité — Principes et fondements* (Volume 1) and *Agencity — Advanced Mathematical Foundations and Extensions* (Volume 2). Git history is migration context only; it does not define canonical physics.

## Status vocabulary

- **canonical theory**: an accepted theoretical definition.
- **maintainer decision**: an explicit project decision resolving an implementation ambiguity or source inconsistency.
- **mathematical consequence**: follows from canonical definitions without an additional physical assumption.
- **numerical approximation**: a numerical realization of a continuous operator.
- **implementation convention**: an explicit software rule needed to realize the theory without redefining it.
- **diagnostic**: interpretation of computed canonical quantities.
- **legacy**: compatibility or historical behaviour outside the reference path.
- **experimental**: an implementation or extension still under investigation.
- **research/speculative**: theoretical extension outside the stable scalar contract.

## Scientific decisions for the reference implementation

The following decisions are explicit and normative for AgencityLab:

```text
w and tau are distinct.
w = tau is only a fallback convention when w is unspecified.

e is Euler's number exp(1).

A_ref is a fixed physical/contextual reference amplitude.

P_c is finite and may be zero: P_c >= 0.

S = 0 implies U = 0 and beta = 0.

Real-agencity is an analysis diagnostic, not a canonical observable.
```

When `w` is omitted, result metadata records:

```text
w was unspecified; implementation convention w = tau was used
```

An explicitly supplied `w` is preserved exactly.

## Source notes and resolved divergences

The source material contains historical layers that must not be silently blended.

- Volume 1 initially presents the CRM using a window tied to `tau`, while its later generalized formulation and Volume 2 use an independent width `w > 0`. The maintainer decision is to treat `w` and `tau` as distinct; `w=tau` is only an omission fallback or an explicit experimental postulate.
- The principal Volume-1 contrast definition and Volume 2 use the fixed number `e = exp(1)`. A later Volume-1 appendix contains historical wording allowing a generic positive offset and mentioning `e=1`. AgencityLab follows the accepted main definition and the explicit maintainer decision: `e` is Euler's number and is not configurable.
- Volume 2 writes its discrete Chapter-12 differences in sampled notation `u_n`, `X_n`, `A_n`. AgencityLab applies those exact stencils to the canonical reduced sequence `u_star` on `delta_star = delta/tau`, preserving the project-wide normalization/reduced-time contract.

These source differences are documented as source history; the code does not alter the accepted theory to make tests pass.

## Reference canonical pipeline

There is one reference end-to-end canonical pipeline:

```text
agencitylab.compute_agencity()
```

It implements:

```text
u* = u / A_ref
t* = t / tau
X* = d(u*) / d(t*)
A* = d^2(u*) / d(t*)^2
M  = CRM_w[u*]
O  = CRM_w[u*, X*]
D  = sqrt((X*)^2 + (A* X*)^2)
S  = sqrt(M^2 + O^2)
J  = ln((e + D) / (e + S))
Theta = atan2(O, M)
U  = (M + i O) / S   if S > 0, else 0
beta = J U            if S > 0, else 0
b = P_c beta
```

Here `e = exp(1)` and `P_c >= 0`. Therefore `P_c = 0` implies `b = 0` exactly, with no division by `P_c` and no epsilon substitution.

`agencitylab.core.agencity.compute_full_agencity()` is retained only as a deprecated **legacy compatibility wrapper**. It delegates to `compute_agencity()` and is not an independent canonical pipeline. Its historical diagnostic payload is also legacy.

## Canonical parameter policy

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual quantities rather than ordinary signal statistics.

- `A_ref > 0` is fixed by physical/contextual reference information and is used exactly in `u* = u/A_ref`. Statistical helpers such as standard deviation, MAD, range, min/max or z-score are not silent canonical fallbacks.
- `tau > 0` is the characteristic structural time. Signal-derived `estimate_tau()` remains non-canonical and is not called by the reference pipeline.
- `w > 0` is the CRM memory width and is independent of `tau`. Omission invokes only the documented fallback `w=tau`.
- `P_c >= 0` is characteristic physical/contextual power. Stable NumPy computation requires finite values. Scalars, sampled profiles, and supported callables may contain exact zeros; negative, NaN, and infinite values are rejected.

Sampling interval, `tau`, CRM width `w`, and multiscale analysis remain distinct concepts.

## Operator mapping

| Theory stage | Implementation | Status |
| --- | --- | --- |
| `u -> u*` | `agencitylab/core/normalization.py` | canonical theory |
| `t -> t*` | `agencitylab/core/activation.py::reduced_coordinate` | canonical theory |
| continuous `X*` | `agencitylab/core/activation.py` | canonical definition + numerical finite-difference approximation |
| continuous `A*` | `agencitylab/core/activity.py` | canonical definition + successive numerical finite-difference approximation |
| Volume-2 discrete `X_n`, `A_n` | `agencitylab/core/discrete.py` | explicit discrete theory |
| CRM | `agencitylab/core/crm.py` | canonical adjacent causal windows; exact zero-variance convention |
| `M`, `O` | `agencitylab/core/memory.py`, `organization.py` | canonical theory |
| `D`, `S` | `agencitylab/core/intensity.py` | canonical theory |
| `J` | `agencitylab/core/contrast.py` | canonical theory; fixed `np.e` |
| `Theta`, `U` | `agencitylab/core/orientation.py` | canonical theory |
| `beta` | `agencitylab/core/beta.py` | canonical theory |
| `b = P_c beta` | `agencitylab/core/agencity.py::agencity` | canonical theory |
| scalar orchestration | `agencitylab/api/compute.py::compute_agencity` | sole reference canonical pipeline |
| real-agencity/coherence interpretation | `agencitylab/analysis/coherence.py` | diagnostic |
| old core coherence/criteria helpers | `agencitylab/core/coherence.py`, `core/agencity.py::agencity_criteria` | legacy diagnostic |

## Continuous sampled pipeline versus Volume-2 discrete formulation

Two legitimate numerical constructions must not be conflated.

### Continuous pipeline sampled numerically

`compute_agencity()` approximates the continuous reduced derivatives through NumPy gradients:

```text
X* ~= gradient(u*)
A* ~= gradient(X*)
```

This is a **numerical approximation of the continuous reference pipeline**.

### Explicit Volume-2 discrete construction

`compute_discrete_agencity()` uses the Chapter-12 interior stencils explicitly:

```text
X_n = (u[n+1] - u[n-1]) / (2 delta)
A_n = (u[n+1] - 2 u[n] + u[n-1]) / delta^2
```

In AgencityLab these are applied to `u_star` with `delta_star=delta/tau`. Result length is preserved with second-order one-sided endpoint formulas; that endpoint choice is an **implementation convention** allowed by the Volume-2 boundary discussion.

For `u(t)=t^2`, the direct second difference returns the exact constant second derivative. For a sinus `sin(omega t)`, the interior transfer factors are

```text
X: omega * sin(z)/z,                  z = omega delta
A: -omega^2 * 4 sin^2(z/2) / z^2
```

with no interior phase shift. Successive centred first differences instead give a different second-derivative amplitude factor `(sin(z)/z)^2`. The two formulations therefore converge to the same continuous limit but are not the same finite-resolution operator.

## Exact null/rest-state convention

For an exactly constant sampled observable, the reference implementation uses the accepted null/rest-state representation:

```text
X* = A* = M = O = D = S = J = 0
U = beta = b = 0
```

The check is exact, not epsilon-based. More generally, wherever `S = 0`:

```text
U = 0
beta = 0
```

Zero agencity is a valid state, not an anomaly.

## Numerical safeguards versus physics

Canonical operators do not insert machine epsilon into physical equations.

- CRM uses exact zero-variance semantics. Its optimized implementation may select a direct centred computation when floating-point cancellation makes rolling moments unreliable.
- `S = hypot(M,O)` is not replaced by `S+EPS`.
- `U` branches on the exact mathematical condition `S > 0`.
- `J` uses `e=exp(1)`, never numerical epsilon.
- `P_c=0` remains zero.
- tiny positive physical values are not globally reclassified as zero.

Legacy diagnostic utilities may use a numerical floor internally to keep diagnostic functions such as a logarithmic circular standard deviation finite. Such safeguards are outside canonical physics.

## Interpretation boundary

The core computes mathematical quantities. It does not decide whether a system has coherent or “real” agencity.

`beta != 0` alone is not a real-agencity criterion. The modern reference diagnostic layer is `agencitylab/analysis/`, where contextual interpretation may use `S > 0`, angular stability or `Sigma_Theta`, significant `|b|`, and persistence. Diagnostic thresholds are not universal constants and never modify `beta`.

## Conformance test contract

The suite locks at least the following properties:

- explicit `w != tau` is accepted and preserved;
- omitted `w` records the implementation fallback `w=tau`;
- canonical `A_ref` is fixed/contextual and has no statistical signal fallback;
- `J` uses Euler's number `exp(1)` rather than `1` or epsilon;
- finite `P_c=0` is accepted for scalar and sampled power and gives exact zero flux;
- negative and non-finite `P_c` are rejected;
- `S=0 => U=0 => beta=0` exactly;
- the public result carries no real-agencity classification;
- the historical full-pipeline helper delegates to the sole canonical reference path;
- Volume-2 discrete stencils are tested on constant, linear, quadratic, sinusoidal and dynamical signals;
- discrete sinus transfer, boundary behaviour, convergence, and propagation through `D`, `S`, `J`, `beta`, and `b` are tested;
- deterministic scientific reference systems remain regression gates.

These tests establish implementation fidelity and numerical regression coverage. They do not constitute universal empirical validation of the Theory of Agencity.
