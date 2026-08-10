# Theory-to-code mapping

This document records the current relationship between the Agencity theory sources used by the project and the Python implementation. It is a traceability document, not a claim that the code is already scientifically validated.

## Status vocabulary

- **canonical target**: the definition selected from the current main theory text for the reference implementation.
- **experimental**: a deliberate extension or alternative formulation.
- **heuristic**: a practical modelling or robustness rule not promoted to a theoretical axiom.
- **diagnostic**: an analysis quantity used to interpret outputs.
- **legacy**: behaviour inherited from an earlier or divergent formulation and awaiting reconciliation.

## Canonical target pipeline

The current main theory source gives the reduced pipeline, using reduced time `t*` and normalized observable `u*`:

```text
X* = d(u*) / d(t*)
A* = d^2(u*) / d(t*)^2
M  = CRM[u*]
O  = CRM[u*, X*]
D  = sqrt((X*)^2 + (A* X*)^2)
S  = sqrt(M^2 + O^2)
J  = ln((e + D) / (e + S))
U  = (M + i O) / S, for S > 0
beta = J U
b = P_c beta
```

For `S = 0`, the selected theory formulation sets the structured complex contribution to zero rather than assigning an arbitrary orientation.

The main theory text also uses the characteristic scale `tau` in normalization/time reduction and in the causal memory construction. Validation across physical, biological, economic, or computational domains remains a research programme rather than an established package guarantee.

## Current mapping

| Theory stage | Canonical target | Current implementation | Status in 0.1.12 |
| --- | --- | --- | --- |
| observable | `u` | API input preparation in `agencitylab/api/compute.py` | canonical interface target |
| normalization | `u -> u*` | `agencitylab/core/normalization.py` | under scientific review |
| activation | `X* = d(u*)/d(t*)` | `agencitylab/core/activation.py` | under scientific review |
| activity | `A* = d(X*)/d(t*)` | `agencitylab/core/activity.py` | under scientific review |
| characteristic scale | `tau` | `agencitylab/core/tau.py` and API metadata resolution | mixed canonical/heuristic depending on auto-resolution |
| memory | `M = CRM[u*]` | `compute.py` passes `A*` to `core.memory.memory`; `memory.py` applies `tanh(CRM(A*))` | **legacy/divergent** |
| organisation | `O = CRM[u*, X*]` | `compute.py` passes `X*` to `core.organization.organization`; `organization.py` applies `tanh(CRM(X*))` | **legacy/divergent** |
| dynamic intensity | `D = sqrt((X*)^2 + (A*X*)^2)` | `agencitylab/core/intensity.py` | canonical target, verify numerically in v0.2 |
| structural intensity | `S = sqrt(M^2 + O^2)` | `agencitylab/core/intensity.py` | canonical target conditional on corrected `M`, `O` |
| contrast/orientation | logarithmic `J`, complex direction `U` | `agencitylab/core/beta.py` and related core helpers | canonical target, verify edge cases in v0.2 |
| structured Agencity | `beta = J U` | `agencitylab/core/beta.py` | canonical target conditional on upstream stages |
| observable flow | `b = P_c beta` | `agencitylab/core/agencity.py`, characteristic power resolution in `core/power.py` | canonical target; `P_c="auto"` is a separate modelling/API concern |
| interpretation | coherence, regimes, signatures, information diagnostics | `agencitylab/analysis/` | mostly diagnostic/experimental |

## Known discrepancies that must not be hidden

### 1. Memory and organisation operands

The selected current main theory text uses `M = CRM[u*]` and `O = CRM[u*, X*]`. The current pipeline instead computes memory from `A*` and organisation from `X*` as separate single-signal CRM operations.

This is a scientific difference, not a naming issue. Version `0.1.12` documents it but does not silently rewrite the numerical core. Reconciliation belongs in the v0.2 phase with focused tests and an explicit migration decision.

### 2. `tanh` compression

`core/memory.py` and `core/organization.py` currently apply `tanh` to CRM outputs and describe that behaviour as canonical. The selected current main theory formulation used for this mapping does not include that saturation in the definitions of `M` and `O`.

Until reconciled, the `tanh` transform should be treated as legacy/experimental behaviour, not as an unquestioned canonical equation.

### 3. CRM window versus `tau`

The current main theory text uses a causal comparison over adjacent windows tied to `tau`. An advanced theory treatment also studies a more general window `w > 0` and its optimisation. The implementation in `core/crm.py` adds a short-observation rule that switches from `w = tau` to `w = tau / A_fact` when `T_obs < 2*tau` (or when compression is forced).

That compression rule is an implementation modelling choice. It must not be conflated with the canonical main-text definition, with sampling resolution, or with multiscale analysis. The v0.2 reconciliation should decide whether it remains as a labelled heuristic/experimental mode.

### 4. Divergent source formulations

The theory material contains at least one appendix/alternate presentation that uses raw-variable CRM inputs where the main text uses reduced quantities, and it presents a different-looking convention for the logarithmic regularisation constant. These differences should be arbitrated at the theory level before code is changed.

The project therefore does not use "the newest Python implementation" as a source of truth when theory sources disagree.

## Interpretation boundaries

The following cautions are part of the project contract:

- high `D` is dynamic intensity, not proof of agency;
- `S` is structural intensity, not a synonym for dynamic activity;
- `U` is a structural direction/orientation when defined;
- `beta` or `b` alone should not be advertised as a universal agency/noise classifier;
- real-agencity diagnostics that combine structural intensity, orientation stability, and significant `|b|` belong to analysis/validation and do not redefine the core observable;
- stochastic signals need not have `D = 0`.

## v0.2 reconciliation checklist

The next phase should, in order:

1. select and cite the exact canonical definitions for `M` and `O`;
2. add unit tests for those definitions on deterministic synthetic signals;
3. decide the status of `tanh` and preserve it only if explicitly labelled and justified;
4. separate canonical CRM windowing from optional/experimental window policies;
5. verify `D`, `S`, `J`, `U`, `beta`, and `b`, including `S = 0` and numerical-stability cases;
6. update API/docstrings that currently call divergent behaviour canonical;
7. only then expand validation, multiscale optimisation, or classification claims.
