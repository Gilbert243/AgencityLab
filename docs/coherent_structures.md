# Coherent Structures and Field Topology

**Software version:** 1.1.2  
**Scientific status:** `research`

AgencityLab 1.1.2 exposes static/reference coherent structures for the autonomous complex field `phi`. These helpers are mathematical/research references from Volume 2 and do not constitute empirical validation.

## Domain wall reference

Volume 2 gives the normalized real-sector kink

```text
psi(x) = tanh((x - x0) / sqrt(2)).
```

For the shared quartic potential, `domain_wall_profile()` uses the rescaled consequence

```text
v = sqrt(lambda / mu)
phi(x) = v tanh(sqrt(lambda/2) (x - x0)).
```

This general `(lambda, mu)` form is documented as a **mathematical rescaling**, not a second independent source formula.

The wall is deliberately a **real-sector / Z2 reference solution**. The full complex field has a connected U(1) vacuum manifold, so AgencityLab does not present this real kink as a generally stable topological wall of the full U(1) theory.

`domain_wall_residual()` evaluates the same static equation used by the classical Klein-Gordon acceleration:

```text
laplacian(phi) - potential.gradient(phi).
```

The 1.1.2 integration tests verify this cross-layer identity directly.

## U(1) vortex reference

The two-dimensional vortex reference is

```text
phi(r, theta) = v f(r) exp(i n theta),
```

where `n` is an integer winding number and `v = sqrt(lambda/mu)` for `lambda > 0`.

The source does not provide a closed-form exact radial profile. Accordingly, `vortex_field()` requires the radial profile from the caller and `vortex_radial_residual()` evaluates the radial equation without inventing an exact solution or a universal core scale.

## Spatial winding

`phase_winding(phi_contour)` computes the unwrapped phase change around an ordered closed contour divided by `2*pi`. It returns the numerical floating result instead of silently rounding to an integer.

A contour crossing an exact field zero is rejected because the phase is undefined there. `field_zero_mask()` detects exact zeros by default; near-zero detection requires an explicit user-provided tolerance. No universal threshold is introduced.

This spatial topology diagnostic is separate from temporal winding diagnostics of the canonical `beta(t)` trajectory and is not a criterion of real agencity.

## Coupling to dynamics

The coherent-structure package remains independent of the time solvers. In 1.1.2, reference profiles can be supplied explicitly as initial conditions to the classical field simulators, while the structures themselves remain defined and testable without importing `agencitylab.fields.dynamics`.
