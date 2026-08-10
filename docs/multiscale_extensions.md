# v0.6 — Multiscale & Extensions

Version 0.6 made scale studies, explicit memory-window handling, discrete data, and multivariate observables usable from the public API. These are not a second Theory of Agencity: they are definitions and constructions from Volume 2 of the same theory.

## Theoretical boundary

Volume 2 keeps the CRM width `w > 0` distinct from the characteristic structural time `tau`. It often uses the convenient convention `w = tau`, but Chapter 13 treats `w` as a parameter in its own right and defines objective window-selection criteria.

Consequently the scalar public API supports both cases:

```python
# Common convention
result = compute_agencity(..., tau=2.0)

# Explicit theory parameter
result = compute_agencity(..., tau=2.0, w=1.0)
```

When `w` is omitted, AgencityLab sets `w=tau`. When it is supplied, the positive value is preserved exactly and recorded in metadata. No signal statistic is used by `compute_agencity()` to choose `A_ref`, `tau`, `w`, or `P_c`.

## The b(t, tau) spectrum

`compute_agencity_spectrum()` evaluates the scalar equations at each supplied structural scale. By default:

```text
w_k = tau_k
```

and the returned matrix `b` has shape `(n_scales, n_samples)`. The result also contains `beta`, `b_mean`, `b_rms`, `beta_mean`, `J_mean`, and `S_mean`.

Passing `windows=` keeps `w` independent from `tau`. A scalar keeps the same `w` at every `tau`; a sequence supplies one `w_k` per scale. This is a software API for a distinction already present in Volume 2, not an alternative theory.

## Memory-window optimisation

Chapter 13 defines three objective criteria and an automatic selection procedure. AgencityLab currently implements the angular-stability criterion

```text
Phi2(w) = time mean of Var(Theta_w(s); s in [t-w, t])
w_opt = argmin Phi2(w)
```

through `optimize_agencity_window()`. Candidate widths are represented by integer sample counts for uniformly sampled discrete signals because the discrete construction uses `w=N delta`.

A one-sample CRM, or any candidate for which no complete interval has `S > 0`, does not have a defined structural orientation. Such a candidate is marked ineligible rather than turning the `Theta=0` storage convention into an artificial zero angular variance. This is a documented numerical eligibility rule, not a modification of the theoretical functional.

The default candidate search spans one sampling step to half the observation duration on a logarithmic grid, then quantises candidates to integer sample counts. The implementation does not claim uniqueness of the optimum.

## Discrete signals

The theory's discrete construction uses centered finite differences in the interior, one-sided endpoint formulas where needed, and a discrete CRM over adjacent `N`-sample blocks with `w=N delta`.

`compute_discrete_agencity()` is a convenience entry point:

```python
result = compute_discrete_agencity(
    u,
    delta=0.01,
    A_ref=1.0,
    tau=2.0,
    w=1.0,
    P_c=5.0,
)
```

It constructs `xi_n = t0 + n delta` and delegates to `compute_agencity()`. It does not introduce a second set of equations. If `w` is omitted, the common `w=tau` convention is used.

## Multivariate construction

For `u(t) in R^m`, Volume 2 specifies scalar Agencity per component and characteristic power `P_c,k`, followed by

```text
beta_multi(t) = sum_k P_c,k(t) beta_k(t) / sum_k P_c,k(t)
b_total(t)    = sum_k b_k(t)
              = sum_k P_c,k(t) beta_k(t)
```

`compute_multivariate_agencity()` implements this aggregation. `A_ref`, `tau`, and optionally `w` may be scalar or supplied per component. `P_c` may be a scalar, one value per component, or a sampled matrix. Time-varying component power therefore produces pointwise power weighting.

The discarded arithmetic average is not used.

## tau, w, sampling, and multiscale are different objects

- `tau` is the characteristic structural time and controls reduced time and reduced derivatives.
- `w` is the CRM memory-window width.
- sampling interval `delta` is numerical coordinate resolution.
- a multiscale spectrum compares explicitly supplied `tau` values; it is not an estimator of the physical `tau`.

The compatibility helper `find_optimal_tau()` remains a diagnostic selector over a user-supplied scale grid. It does not claim to infer the physical characteristic time.

The finite-record CRM warm-up associated with memory/organisation is governed by `w`: two adjacent CRM windows require an interval of length `2w`. This is distinct from the derived indicator `Sigma_Theta`, whose definition in the complete formulary remains `Var(Theta(s); s in [t-tau,t])`.

## Riemannian construction

Volume 2 defines an intrinsic direction for systems on a Riemannian manifold by replacing ordinary derivatives with covariant derivatives and defining

```text
D = sqrt(||X||_g^2 + g(A, X)^2)
```

It also says that scalar quantities such as speed or `g(A,X)` may feed the CRM, but explicitly defers the detailed analysis. The document therefore does not yet determine a complete production-grade CRM/vector-state construction, validation contract, and numerical discretisation for a general manifold.

For that reason AgencityLab **does not invent the missing details**. `riemannian_extension_status()` reports this part as `experimental` and unimplemented because the source itself leaves the detailed analysis for future work.

## Public entry points

```python
from agencitylab import (
    compute_agencity,
    compute_agencity_spectrum,
    optimize_agencity_window,
    compute_discrete_agencity,
    compute_multivariate_agencity,
    riemannian_extension_status,
)
```

## What this layer does not claim

A spectrum maximum is not automatically the physical `tau`. An optimised `w` is selected from the signal by a theory-defined criterion and should still be reported as such when the physical context does not independently specify a window. Pc-weighted multivariate aggregation applies the theory's componentwise construction; it does not invent a coupled-vector CRM beyond what the source defines. The Riemannian construction remains a documented research boundary until the missing details are sufficiently specified and testable.
