# Biological Agencity

**Scientific status: experimental mapping of the Theory of Agencity.**

`agencitylab.biology` implements the biological mapping defined by the
*Foundations of Biological Agencity* without changing the canonical Agencity
equations. AgencityLab implements the theory so it can be used reproducibly; it
does not decide whether the theory is empirically true or false.

## Reference structure

A biological protocol fixes a common reference `(R, s)`, where `R` records the
biological support, a causal calibration port, external reference conditions and
a prespecified applicability domain. Individual characteristic capacity `P_c`
remains distinct from the common class/reference quantities `tau(R,s)` and
`w(R,s)`.

```python
import agencitylab.biology as bio

reference = bio.BiologicalReference(
    taxon="example species",
    support="whole organism",
    causal_input="standardized challenge",
    causal_output="reference response",
    identifier="example-reference",
)
```

## Characteristic time

For standardized sustainable powers `P_std` in watts and masses in kilograms,
Biology implements

```text
q_ref = median(P_std / mass)
tau(R,s) = (1 J kg^-1) / q_ref
```

No signal statistic is used to invent `tau`.

## CRM depth

For an independently identified impulse response `h(t)`, the causal time is

```text
theta = integral(t |h(t)| dt) / integral(|h(t)| dt)
```

and the common reference memory depth is the population mean of those causal
times. `tau` and `w` remain distinct quantities even if they happen to be
numerically equal in one protocol. The biological workflow never silently uses
`w=tau`.

## Observable and metrology

The reconstructed biological observable is organized as

```text
u = (Q, Gamma, Delta, Psi, Phi)
```

with a variable number of coordinates in each block. Instrument channels are
observations `Z`, not coordinates of `u` by declaration. A calibrated
measurement map must reconstruct the observable explicitly. The built-in
`BiologicalMeasurementMap` provides a simple versioned affine map with explicit
input units; specialized
metrology can construct `BiologicalObservable` directly while preserving the
same provenance boundary. Measurement covariance is propagated when it is
explicitly supplied; per-channel uncertainty alone is not silently treated as
independent covariance.

Every coordinate has its own positive `A_ref` with the same unit as that
coordinate. Biology does not derive `A_ref` from cohort standard deviation,
z-scores, signal ranges or target-separation performance.

## Frozen protocols

`BiologicalProtocol` records the reference, structural epoch capacity, `tau`,
`w`, observable coordinates, measurement-map identity and dataset roles.
`freeze_protocol()` validates the mapping and produces a SHA-256 addressed
snapshot. Frozen protocols have serialization helpers that verify the hash on
reconstruction. Discovery/closure/calibration/test sample identifiers are kept
as explicit non-overlapping roles.

`BiologicalReferenceRegistry` is intentionally empty by default. It can hold
source-backed, versioned `(R,s)` bundles, but it never invents or silently
selects a biological reference version.

## Computation

`compute_biological_agencity()` is intentionally thin. For one coordinate it
delegates to the public canonical `compute_agencity()` API. For multiple
coordinates it delegates to `compute_multivariate_agencity()` only when the
caller supplies an explicit physical component partition `P_c,k` whose sum is
the organism `P_c`.

The organism capacity is **not** silently repeated once per observable
coordinate. When no physically justified component partition is available, the
multivariate aggregate is left unresolved instead of inventing one.

## Scope boundary

The module does not provide medical diagnosis, tumor-specific reference scales,
or target-dependent calibration. Tumor or other outcome labels belong downstream
of the frozen reference protocol. Population statistics remain analysis products;
a cohort is not automatically a composite physical system, and a collection of
subjects is not automatically a spatial Agencity field.
