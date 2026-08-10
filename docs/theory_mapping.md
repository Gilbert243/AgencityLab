# Theory-to-code mapping

This document records the relationship between the current Agencity theory and the Python reference implementation. The scientific source of truth is the current theory material, especially *Théorie de l’Agencité — Principes et fondements* (second edition, 10 August 2026). Git history is useful for migration context only; it does not define canonical physics.

## Status vocabulary

- **canonical**: directly implements the selected current theory definition.
- **mathematical consequence**: follows from canonical definitions without an additional physical assumption.
- **numerical approximation**: discrete approximation of a canonical continuous operator.
- **implementation convention**: software rule required to make the discrete implementation explicit without redefining the physics.
- **diagnostic**: interpretation layer that does not redefine the observable.
- **experimental**: deliberate extension or alternative formulation.
- **legacy**: retained compatibility or historical behaviour outside the canonical path.

## Canonical pipeline implemented in 0.2.0

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

Here `e = exp(1)`. No `tanh` saturation is part of the second-edition canonical construction.

## Canonical parameter policy

`A_ref`, `tau`, `w`, and `P_c` are not signal statistics.

- `A_ref` is a physical/contextual reference amplitude. It must be supplied explicitly, carried by metadata, or resolved from a deliberately registered physical convention. Canonical normalization never estimates it from standard deviation, MAD, range, or z-score.
- `tau` is a structural characteristic time of the system. It must be supplied explicitly, carried by metadata, or resolved from a deliberately registered physical convention. Signal-derived `estimate_tau` remains an experimental helper and is not used by the canonical pipeline.
- `w` is fixed to `tau` in the v0.2 canonical public pipeline. The advanced mathematical treatment of independently optimized `w` is an extension and does not silently replace this rule.
- `P_c` is a characteristic physical power. It must be supplied explicitly, carried by metadata, derived from documented physical energetics, or resolved from a deliberately registered physical convention. Signal-derived power estimators remain experimental helpers.

Sampling interval, `tau`, CRM window `w`, and multiscale analysis are distinct concepts.

## Operator mapping

| Theory stage | Implementation | v0.2.0 status |
| --- | --- | --- |
| `u -> u*` | `agencitylab/core/normalization.py` | canonical |
| `t -> t*` | `agencitylab/core/activation.py::reduced_coordinate` | canonical |
| `X*` | `agencitylab/core/activation.py` | canonical continuous definition, finite-difference numerical approximation |
| `A*` | `agencitylab/core/activity.py` | canonical continuous definition, finite-difference numerical approximation |
| CRM | `agencitylab/core/crm.py` | canonical adjacent causal windows; exact zero-variance convention |
| `M = CRM[u*]` | `agencitylab/core/memory.py` | canonical |
| `O = CRM[u*, X*]` | `agencitylab/core/organization.py` | canonical |
| `D` and `S` | `agencitylab/core/intensity.py` | canonical |
| `J` | `agencitylab/core/contrast.py` | canonical |
| `Theta`, `U` | `agencitylab/core/coherence.py`, `orientation.py` | canonical |
| `beta` | `agencitylab/core/beta.py` | canonical |
| `b = P_c beta` | `agencitylab/core/agencity.py` | canonical |
| public `u -> b` orchestration | `agencitylab/api/compute.py` | canonical reference path |

## Exact null/rest-state postulate

For the reference implementation, an exactly constant sampled observable is treated as the canonical null/rest-state postulate rather than as something to be numerically proved by finite differences.

The pipeline therefore performs an exact preliminary check, with no tolerance or epsilon. If all sampled values of `u` are exactly identical, derivative and CRM stages are bypassed and the implementation sets

```text
X* = A* = M = O = D = S = J = 0
U = beta = b = 0
```

exactly.

This rule prevents floating-point derivative residue from being mistaken for physical dynamics. It is not a universal threshold for near-constant signals: arbitrarily small but non-zero structure is still processed by the canonical equations.

## Numerical safeguards versus physics

The canonical operators do not insert machine epsilon into valid physical equations.

- Pearson CRM returns zero only when an empirical variance is exactly zero.
- `S = sqrt(M^2 + O^2)` uses the direct norm.
- `U` branches explicitly on `S > 0`; when `S = 0`, `U = 0`.
- `J = ln((e + D)/(e + S))` uses the fixed theoretical constant `e`, not a numerical epsilon.
- tiny positive values are not reclassified as zero by a global threshold.

Generic safeguard utilities may still exist for non-canonical or legacy code, but the canonical `compute_agencity` path does not use them to alter `A_ref`, `tau`, CRM, `D`, `S`, `J`, `U`, `beta`, or `b`.

## Historical migration resolved in v0.2

Version 0.1.12 documented several legacy divergences. Version 0.2.0 removes them from the canonical path:

- historical `tanh` compression of `M` and `O` is no longer used;
- `M` is no longer computed from `A*`;
- `O` is no longer a single-signal CRM of `X*`;
- short-observation `tau / A_fact` CRM compression no longer modifies the canonical window;
- signal statistics are no longer used as silent fallbacks for `A_ref`, `tau`, or `P_c`;
- smoothing, clipping, and saturation are rejected by canonical operators when requested through the canonical API.

Legacy or experimental helpers may remain for comparison or research, but they are outside the canonical reference computation.

## Interpretation boundary

The core computes the observable. It does not decide whether a system has coherent or "real" agencity.

`beta != 0` alone is not a classification criterion. Coherence, angular variance, significant `|b|`, regimes, geometry, events, transitions, and related interpretation belong to `agencitylab/analysis/` or clearly labelled diagnostic functions. Noise and chaos may produce local non-zero `beta`; high `D` is not proof of real agencity.

## v0.2 analytical test contract

The test suite checks the mathematical operators rather than preferred scientific outcomes. It includes:

- exact physical normalization by `A_ref`;
- reduced-time derivatives on an analytic polynomial;
- exact rest-state short-circuit before derivative/CRM evaluation;
- repeating, inverted, cross, zero-variance, and tiny-amplitude CRM cases;
- `M` and `O` canonical operands;
- exact `D` and `S` norms;
- exact logarithmic contrast and `D = S => J = 0`;
- explicit `S = 0` orientation/beta branch;
- tiny positive `S` values below common epsilons to prove that numerical epsilon does not redefine physical zero;
- linearity of `b` in `P_c`;
- rejection of arbitrary physical-parameter fallbacks and non-canonical `w != tau`.

These tests establish implementation fidelity to the selected formulas and conventions. They are not empirical validation of the Theory of Agencity across physical domains.
