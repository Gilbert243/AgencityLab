# Theory-to-code mapping

This document records the relationship between the accepted Agencity theory and the Python reference implementation. The scientific source of truth is the project theory material, especially *Théorie de l’Agencité — Principes et fondements* and *Agencity — Advanced Mathematical Foundations and Extensions*. Git history is migration context only; it does not define canonical physics.

## Status vocabulary

- **canonical**: directly implements an accepted theory definition.
- **mathematical consequence**: follows from canonical definitions without an additional physical assumption.
- **numerical approximation**: discrete approximation of a canonical continuous operator.
- **implementation convention**: explicit software rule required to realize the theory numerically without redefining it.
- **diagnostic**: interpretation layer that does not redefine the observable.
- **experimental**: deliberate extension or implementation path under investigation.
- **research/speculative**: theoretical extension not included in the stable scalar software contract.
- **legacy**: retained compatibility or historical behaviour outside the canonical path.

## Canonical pipeline

```text
u* = u / A_ref
t* = t / tau
X* = d(u*) / d(t*)
A* = d^2(u*) / d(t*)^2
M  = CRM[u*]
O  = CRM[u*, X*]
D  = sqrt((X*)^2 + (A* X*)^2)
S  = sqrt(M^2 + O^2)
J  = ln((e + D) / (e + S))
Theta = atan2(O, M)
U  = (M + i O) / S   if S > 0, else 0
beta = J U            if S > 0, else 0
b = P_c beta
```

Here `e = exp(1)`. No `tanh` saturation or epsilon-modified denominator is part of the canonical construction.

## Canonical parameter policy

`A_ref`, `tau`, `w`, and `P_c` are physical/contextual quantities rather than ordinary signal statistics.

- `A_ref` is the physical/contextual reference amplitude. The stable compute path never estimates it from standard deviation, MAD, range, or z-score.
- `tau` is the characteristic structural time. Signal-derived `estimate_tau()` remains an explicitly non-canonical helper and is not used by the stable compute path.
- `w` is the CRM memory width. Volume 2 keeps `w > 0` distinct from `tau`; the stable API preserves an explicit `w`. If `w` is omitted, the implementation convention is `w=tau`. This convention is not a universal identity and not signal-derived optimization.
- `P_c` is characteristic physical/contextual power. It may be supplied as a positive scalar, sampled profile, or callable under the public contract and is never inferred from `u`.

Sampling interval, `tau`, CRM window `w`, and multiscale analysis are distinct concepts.

## Operator mapping

| Theory stage | Implementation | v1.0 status |
| --- | --- | --- |
| `u -> u*` | `agencitylab/core/normalization.py` | canonical |
| `t -> t*` | `agencitylab/core/activation.py::reduced_coordinate` | canonical |
| `X*` | `agencitylab/core/activation.py` | canonical continuous definition, finite-difference numerical approximation |
| `A*` | `agencitylab/core/activity.py` | canonical continuous definition, finite-difference numerical approximation |
| CRM | `agencitylab/core/crm.py` | canonical adjacent causal windows; exact zero-variance convention; optimized implementation preserves the centred Pearson definition |
| `M = CRM[u*]` | `agencitylab/core/memory.py` | canonical |
| `O = CRM[u*, X*]` | `agencitylab/core/organization.py` | canonical |
| `D`, `S` | `agencitylab/core/intensity.py` | canonical |
| `J` | `agencitylab/core/contrast.py` | canonical |
| `Theta`, `U` | `agencitylab/core/orientation.py` and related coherence helpers | canonical orientation |
| `beta` | `agencitylab/core/beta.py` | canonical |
| `b = P_c beta` | `agencitylab/core/agencity.py` | canonical |
| public scalar orchestration | `agencitylab/api/compute.py` | stable canonical reference path |

## Exact null/rest-state convention

For the reference implementation, an exactly constant sampled observable is treated as the accepted null/rest-state convention before derivative or CRM evaluation.

The check is exact, with no tolerance or epsilon. If all sampled values of `u` are identical, the implementation sets

```text
X* = A* = M = O = D = S = J = 0
U = beta = b = 0
```

and bypasses derivative/CRM operators. Near-constant or tiny-amplitude non-constant signals still run through the canonical equations.

## Numerical safeguards versus physics

Canonical operators do not insert machine epsilon into valid physical equations.

- CRM uses the exact zero-variance convention; optimized rolling calculations may fall back to a direct centred Pearson calculation when machine cancellation is unsafe.
- `S = sqrt(M^2 + O^2)` uses the direct norm.
- `U` branches explicitly on `S > 0`; at `S = 0`, `U = 0`.
- `J = ln((e + D)/(e + S))` uses the fixed theoretical constant `e`, not numerical epsilon.
- tiny positive values are not globally reclassified as physical zero.

Generic safeguard utilities may exist elsewhere, but the stable `compute_agencity()` path does not use them to alter valid canonical equations.

## Historical migration

The 0.x series progressively removed legacy divergences from the reference path:

- historical `tanh` compression of `M` and `O` is no longer used;
- `M` uses the accepted CRM operand and `O` the accepted cross-correlation operand;
- historical `tau / A_fact` CRM compression does not modify the canonical window;
- Volume 2's distinction between `w` and `tau` is honored by the stable public API since v0.7;
- signal statistics are not silent fallbacks for `A_ref`, `tau`, `w`, or `P_c`;
- smoothing, clipping, or saturation are not inserted into canonical operators by the stable API.

Legacy or experimental helpers may remain for comparison or research, but they are outside the stable reference computation.

## Interpretation boundary

The core computes the observable. It does not decide whether a system has coherent or real agencity.

`beta != 0` alone is not a classification criterion. Coherence, angular variance, significant `|b|`, persistence, regimes, geometry, events, transitions, and related interpretation belong to `agencitylab/analysis/` or clearly labelled diagnostic functions. Noise and chaos may produce local non-zero `beta`; high `D` is not proof of real agencity.

## v1.0 test contract

The release suite checks mathematical operators and accepted reference consequences rather than preferred outcomes. It includes:

- exact physical normalization by `A_ref` and reduced-time derivatives;
- exact rest-state short-circuit;
- repeating, inverted, cross, zero-variance, tiny-amplitude, and optimized/direct-equivalence CRM cases;
- canonical `M`, `O`, `D`, `S`, `J`, `Theta/U`, `beta`, and `b` identities;
- explicit `S = 0` and `D = S` branches;
- tiny positive structure below common machine epsilons;
- exact linearity of `b` in `P_c`;
- explicit positive `w` values distinct from `tau` plus the omission convention `w=tau`;
- deterministic scientific reference systems and theory-property tests;
- batch/thread, full-history streaming/one-shot, and multiscale/scalar equivalence;
- stable result serialization/export and packaging workflows.

These tests establish implementation fidelity and numerical regression coverage. They are not empirical validation of the Theory of Agencity across physical domains.
