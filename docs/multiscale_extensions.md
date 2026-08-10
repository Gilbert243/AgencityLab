# v0.6 — Multiscale & Extensions

Version 0.6 turns the scalar Agencity reference implementation into an explicit extension layer for scale studies, discrete data, and multivariate observables while preserving the v0.2–v0.5 canonical path.

## Scientific boundary

The stable scalar API remains canonical for this project:

```text
compute_agencity(..., tau=tau, w=tau, ...)
```

It still rejects `w != tau`. This is deliberate. The advanced mathematical volume keeps the CRM width `w > 0` distinct from the characteristic structural time `tau`, often sets `w = tau`, and studies independent `w` when optimising the memory window. v0.6 therefore implements independent `w` only through explicitly named extension APIs.

No signal statistic is used to infer `A_ref`, `tau`, or `P_c` in these APIs.

## The b(t, tau) spectrum

`compute_agencity_spectrum()` evaluates the scalar equations at each supplied structural scale. By default:

```text
w_k = tau_k
```

and the returned matrix `b` has shape `(n_scales, n_samples)`. The result also contains `beta`, `b_mean`, `b_rms`, `beta_mean`, `J_mean`, and `S_mean`.

Passing `windows=` makes `w` an independent explicit coordinate. A scalar keeps the same `w` at every `tau`; a sequence supplies one `w_k` per scale. This is an advanced-theory extension, not a silent change to canonical `compute_agencity()`.

## Memory-window optimisation

The advanced volume defines three objective criteria and gives its automatic algorithm in terms of the angular-stability functional

```text
Phi2(w) = time mean of Var(Theta_w(s); s in [t-w, t])
w_opt = argmin Phi2(w)
```

`optimize_agencity_window()` implements this criterion on the discrete CRM. Candidate widths are represented by integer sample counts because `w = N delta` in the discrete theory.

A one-sample CRM, or any candidate for which no complete interval has `S > 0`, does not have a physically defined structural orientation. v0.6 marks such a candidate ineligible instead of converting the `Theta=0` storage convention into artificial zero angular variance. This is an operational numerical rule, documented in the returned result.

The default candidate search spans the sampled interval from one sampling step to half the observation duration on a logarithmic grid, then quantises candidates to integer sample counts. The algorithm does not claim uniqueness of the optimum.

## Discrete signals

The theory's discrete construction uses centered finite differences in the interior, one-sided endpoint formulas where needed, and a discrete CRM over adjacent `N`-sample blocks with `w=N delta`.

The existing scalar engine already implements this sampled-data path. v0.6 adds `compute_discrete_agencity()` as a convenience entry point:

```python
result = compute_discrete_agencity(
    u,
    delta=0.01,
    A_ref=1.0,
    tau=2.0,
    P_c=5.0,
)
```

It constructs `xi_n = t0 + n delta` and delegates to `compute_agencity()`. It does not introduce a second set of canonical equations.

## Multivariate extension

For `u(t) in R^m`, the advanced volume specifies scalar Agencity per component and characteristic power `P_c,k`, followed by

```text
beta_multi(t) = sum_k P_c,k(t) beta_k(t) / sum_k P_c,k(t)
b_total(t)    = sum_k b_k(t)
              = sum_k P_c,k(t) beta_k(t)
```

`compute_multivariate_agencity()` implements exactly this aggregation. `A_ref`, `tau`, and optionally `w` may be scalar or supplied per component. `P_c` may be a scalar, one value per component, or a sampled matrix. Time-varying component power therefore produces pointwise power weighting.

The earlier arithmetic mean is not used.

## tau, w, and multiscale analysis are different objects

- `tau` is a characteristic structural time and controls reduced time and reduced derivatives.
- `w` is the CRM memory-window width.
- sampling interval `delta` is a numerical coordinate resolution.
- a multiscale spectrum compares explicitly supplied `tau` values; it is not an estimator of the physical `tau`.

The compatibility helper `find_optimal_tau()` remains a diagnostic selector over a user-supplied scale grid. It does not claim to infer the canonical physical parameter.

## Riemannian extension

The advanced source sketches an intrinsic extension on a Riemannian manifold by replacing ordinary derivatives with covariant derivatives and defining

```text
D = sqrt(||X||_g^2 + g(A, X)^2)
```

It also says that scalar quantities such as speed or `g(A,X)` may feed the CRM, but explicitly defers the detailed analysis. The source does not fully determine a production-grade multivariate CRM, structural pair `(M,O)`, validation contract, or numerical discretisation on a general manifold.

For that reason v0.6 **does not invent a Riemannian pipeline**. `riemannian_extension_status()` reports it as `experimental` and unimplemented. A future implementation should wait for sufficiently accepted definitions and coordinate-invariance tests.

## Public entry points

```python
from agencitylab import (
    compute_agencity_spectrum,
    optimize_agencity_window,
    compute_discrete_agencity,
    compute_multivariate_agencity,
    riemannian_extension_status,
)
```

## What v0.6 does not claim

A spectrum maximum is not automatically the physical `tau`. An optimised `w` is signal-derived by an advanced-theory criterion and must not be confused with a separately known characteristic structural time. Pc-weighted multivariate aggregation is an extension of independently computed scalar components, not a new coupled-vector CRM theory. The Riemannian construction remains an explicit research boundary.
