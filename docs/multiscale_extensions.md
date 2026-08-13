# Multiscale & Extensions

These APIs expose scale studies, explicit CRM-window handling, discrete data, and multivariate observables from Volume 2. They do not create a second scalar reference theory.

## `tau` and `w`

Volume 2 keeps CRM width `w > 0` distinct from characteristic structural time `tau`. AgencityLab follows that distinction.

```python
# w omitted: implementation fallback w=tau
result = compute_agencity(..., tau=2.0)

# explicit independent width
result = compute_agencity(..., tau=2.0, w=1.0)
```

When `w` is omitted, metadata records that the implementation fallback was used. When supplied, `w` is preserved exactly. No signal statistic is used by `compute_agencity()` to choose `A_ref`, `tau`, `w`, or `P_c`.

## The `b(t,tau)` spectrum

`compute_agencity_spectrum()` evaluates the scalar equations at each supplied structural scale. With `windows=None`, each row uses the explicit fallback convention `w_k=tau_k`. Passing `windows=` keeps the quantities independent.

A multiscale grid is not an estimator of the physical `tau`; it is an explicit comparison across supplied scales.

## Memory-window optimisation

`optimize_agencity_window()` implements the Chapter-13 angular-stability study over explicit candidate widths. Because the discrete CRM requires `w=N delta`, candidates are represented by integer sample counts.

A candidate with no complete interval on which `S>0` has undefined structural orientation and is excluded rather than using the stored `Theta=0` representation as artificial coherence. Window optimisation is an explicit signal-derived diagnostic/experimental procedure, not silent physical parameter inference.

## Discrete signals: explicit Volume-2 formulation

`compute_discrete_agencity()` implements the Volume-2 sampled construction. It is **not** an alias for `compute_agencity()` and does not silently use `gradient -> gradient` for the second derivative.

For interior samples Volume 2 defines

```text
X_n = (u[n+1] - u[n-1]) / (2 delta)
A_n = (u[n+1] - 2u[n] + u[n-1]) / delta^2
```

AgencityLab applies these operators to the canonical normalized sequence `u_star` on reduced spacing `delta_star=delta/tau`. To preserve the stable full-length result contract, endpoints use explicit second-order one-sided finite differences. This endpoint rule is an implementation convention consistent with the source's allowance for one-sided boundary treatment.

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

After the explicit `X_star` and `A_star` stages, CRM, `M`, `O`, `D`, `S`, `J`, `U`, `beta`, and `b` use the same reference core operators.

### Why this differs from successive gradients

For a sinus `sin(omega t)` and `z=omega delta`, the direct Volume-2 interior operators have transfer factors

```text
X:  omega sin(z)/z
A: -omega^2 4 sin^2(z/2)/z^2
```

with no interior phase shift. Two successive centered first differences instead give a second-derivative amplitude factor `-omega^2 (sin(z)/z)^2`. Both converge to the continuous derivative as `delta -> 0`, but they are distinct finite-resolution operators.

The test suite therefore checks constant, linear, quadratic, sinusoidal, damped, Van der Pol, unstable, and filtered stochastic signals, as well as boundary behaviour and convergence.

## Multivariate construction

For `u(t) in R^m`, Volume 2 specifies scalar Agencity per component and characteristic power `P_c,k >= 0`, followed by

```text
b_total(t) = sum_k P_c,k(t) beta_k(t)
```

When total component power is positive,

```text
beta_multi(t) = b_total(t) / sum_k P_c,k(t)
```

`P_c,k=0` is valid. If every component has zero power at one sample, then `b_total=0` exactly but the weighted mean is mathematically undefined. The stable array stores `beta_multi=0` at that sample and exposes `beta_multi_defined=False`; no epsilon is inserted into the denominator.

`A_ref`, `tau`, and optionally `w` may be scalar or supplied per component. `P_c` may be scalar, one value per component, or a sampled matrix.

## `tau`, `w`, sampling and multiscale are different objects

- `tau`: characteristic structural time and reduced-time scale;
- `w`: CRM memory width;
- `delta`: numerical sampling interval;
- multiscale scan: comparison over explicit `tau` values.

The finite-record CRM history requirement is governed by `w`, while `Sigma_Theta` remains a diagnostic whose time interval is defined separately.

## Riemannian construction

Volume 2 gives a geometric direction using covariant derivatives and an intrinsic dynamic intensity, but defers enough detail that a complete production CRM/vector-state implementation is not yet determined. AgencityLab therefore keeps `riemannian_extension_status()` experimental and does not invent the missing theory.

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

## Scientific boundary

A spectrum maximum is not automatically physical `tau`. An optimised `w` remains a signal-derived selection unless independently justified by context. Pc-weighted multivariate aggregation does not invent a coupled-vector CRM. Riemannian and broader fundamental extensions remain research boundaries.
