# Agencity Analysis — v0.5

Version 0.5 turns the canonical output of `compute_agencity()` into a scientific-analysis layer without changing the canonical computation.

## Layer boundary

The canonical engine remains responsible for

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,Theta -> beta -> b
```

The analysis layer consumes those arrays. It may compute derived indicators, finite-record diagnostics, explicit threshold-based classifications, and reports. It must not alter `A_ref`, `tau`, `w`, `P_c`, CRM, `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b`.

The v0.5 analysis schema is `0.5`. The `AgencityResult` serialization schema remains `0.3` because the canonical result model is unchanged.

## Sigma_Theta

The accepted theory defines the local angular variance as

```text
Sigma_Theta(t) = Var(Theta(s); s in [t - tau, t]).
```

`agencitylab.analysis.sigma_theta()` implements this as an ordinary variance after locally unwrapping the canonical structural orientation inside each complete time window. A value is left undefined (`NaN`) if the complete interval is unavailable or if structural orientation is undefined anywhere in that interval (`S = 0`).

Circular variance `1 - R` remains available as a separate circular-statistics diagnostic. It is not silently substituted for the theoretical `Sigma_Theta`.

Structural coherence uses canonical `Theta = atan2(O, M)`. It does not use `arg(beta)` or `arg(b)` as a substitute, because a negative `J` can rotate `beta = J exp(i Theta)` by pi while leaving the structural direction unchanged.

## Real-agencity criterion

The theoretical criterion is

```text
S > 0
Sigma_Theta low
|b| significant
```

The theory does not provide universal numerical values for `low` or `significant`. Therefore v0.5 has no universal defaults for those two decisions.

Without explicit contextual thresholds, the real-agencity diagnostic returns `status="undetermined"`. With explicit thresholds it returns a local Boolean mask and the fraction of evaluated samples satisfying the criterion. A whole-record Boolean requires an additional explicit `min_fraction`; this prevents one intermittent sample in noise or chaos from being promoted to a global real-agencity claim.

Example:

```python
analysis = analyze_agencity(
    result,
    real_agencity_thresholds={
        "theta_variance_threshold": 0.1,
        "b_threshold": 2.0,
        "min_fraction": 0.7,
    },
)
```

Those numbers are experiment-specific diagnostic settings, not constants of the theory.

## Geometry of beta

The theory defines geometric observables on the intrinsic state curve `beta(t)`, not on `b(t)`. This distinction is essential when `P_c(t)` varies.

The signed algebraic curvature is approximated from

```text
kappa(t) = Im(conj(beta_dot) * beta_ddot) / |beta_dot|^3
```

where defined. If the discrete velocity is exactly zero, curvature is undefined and v0.5 returns `NaN`; no epsilon is inserted into the denominator.

The structural winding diagnostic is

```text
W = (1 / 2*pi) integral Theta_dot dt.
```

A complete closed cycle can give an integer winding number. For a general finite interval, v0.5 reports the raw net winding and its residual from the nearest integer rather than forcing quantization. If `S = 0` inside the analysed topological interval, winding is left undefined rather than bridging the missing structural direction.

## Zeros and transitions

For strictly positive characteristic power,

```text
b = 0 <=> beta = 0
beta = 0 if S = 0
beta = 0 if S > 0 and J = 0
J = 0 <=> D = S
```

`detect_agencity_zeros()` therefore uses `S = 0` or `J = 0` exactly by default. A caller may request a positive numerical tolerance, but that tolerance is explicitly diagnostic and does not redefine physical zero.

`critical_surface_crossings()` detects exact `D = S` samples and sign changes of `D - S`. `detect_theta_jumps()` detects wrapped orientation jumps only when an explicit angular threshold is supplied.

For CRM-dependent finite-record geometry and transition summaries, the high-level report starts at `t >= t0 + 2*tau`, when two complete causal CRM windows are available. This is a numerical boundary convention, not new physics.

## D peaks and S plateaus

`detect_dynamic_peaks()` finds local maxima of the already-computed canonical `D`. Optional prominence and separation filters are explicit diagnostic settings.

`detect_structural_plateaus()` identifies intervals with approximately flat `S` only when both a slope tolerance and a minimum duration are supplied. No universal plateau threshold is provided.

## Regime signatures and classification

`regime_signature(result)` extracts threshold-free finite-record observations such as mean and variance of `|b|`, mean `D/S/J`, tail behaviour, `Sigma_Theta`, beta curvature, a tau-periodicity score, growth ratio, and zero density.

Automatic classification is deliberately separate. The qualitative theory table distinguishes:

- `null`
- `passive_damped`
- `active_oscillating`
- `unstable`
- `stochastic`
- `chaotic`

Only the exact null state can be classified without interpretive thresholds. Non-null records default to `undetermined`. To request automatic classification, supply a `RegimeCriteria` object or equivalent mapping. Its values are contextual diagnostic criteria and are recorded alongside the result.

This conservative design is intentional: noise and chaos may have local non-zero `beta`, and high `D` does not imply coherent agencity.

## Multiscale signatures

The historical log-log signature remains available through `agencity_signature()`. v0.5 no longer replaces zero or invalid scale data with epsilon before logarithms. At least two strictly positive finite pairs are required.

A fitted slope is returned as a numerical observation. The labels `amplifying`, `dissipative`, or `approximately_scale_invariant` are produced only when a caller explicitly supplies a `slope_threshold`.

## Public API

Useful entry points include:

```python
from agencitylab import (
    analyze_agencity,
    analyze_coherence,
    analyze_geometry,
    analyze_events,
    analyze_transitions,
    analyze_regime_signature,
    analyze_regime,
    RegimeCriteria,
)
```

`analyze_agencity(result)` is conservative by default: it computes theory-derived indicators but leaves threshold-dependent real-agencity and non-null regime classification undecided until context is supplied.

## Scientific status

These diagnostics transform `b(t)` and its canonical decomposition into an analysis tool. They do not constitute empirical confirmation that Agencity is a universal physical observable, and their configurable thresholds must not be presented as universal laws.
