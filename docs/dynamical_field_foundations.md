# Dynamical Agencity Field Foundations

**Software version:** 1.1.2  
**Scientific status:** `research` for autonomous `phi` physics and classical dynamics; `experimental` for generic field numerics.

AgencityLab 1.1.2 builds on the 1.1.1 field foundations and now exposes the classical autonomous-field equations and coherent-structure references described in Volume 2. These remain research extensions, not canonical observable definitions and not empirical validation.

## Observable field and dynamical field are distinct

The observable spatial API remains

```text
u(x,t) -> local canonical pipeline -> beta_obs(x,t), b_obs(x,t)
```

and keeps scientific status `experimental` because the spatial orchestration extends the canonical scalar construction without redefining it.

The autonomous research field is introduced only through an explicit bridge:

```text
beta_obs(x,t) -- explicit research bridge --> phi(x,t)
```

with

```text
phi = sqrt(P_c * tau) * beta
```

`compute_agencity_field()` does not perform this promotion automatically. Local `P_c = 0` gives exact `phi = 0`; no epsilon or inverse convention is inserted.

## Quartic reference potential

The reference research potential implemented by `QuarticAgencityPotential` is

```text
V(phi) = -(lambda/2) |phi|^2 + (mu/4) |phi|^4,  mu > 0
```

with source-term convention

```text
g(phi) = -lambda phi + mu |phi|^2 phi.
```

For `phi = x + i y`, this implemented source term equals `dV/dx + i dV/dy`; under the standard Wirtinger normalization it is `2 dV/d(phi*)`. AgencityLab preserves the Volume-2 source-term convention rather than silently dividing it by two.

For `lambda > 0`, the broken-symmetry vacuum amplitude is

```text
v = sqrt(lambda / mu)
```

and `vacuum_state(..., theta=...)` constructs `v exp(i theta)` without privileging a phase.

## Energy primitives

In the chosen dimensionless/natural-unit convention, the research field-energy density is

```text
rho = 1/2 |phi_dot|^2 + 1/2 |grad phi|^2 + V(phi).
```

Physics receives the squared gradient norm as an input. It does not compute spatial derivatives itself.

## Generic numerical infrastructure

`agencitylab.fields.numerics` supplies NumPy-only reusable infrastructure:

- uniform rectilinear N-D grids;
- periodic, Dirichlet, and Neumann boundaries;
- second-order gradient and Laplacian operators;
- squared gradient norm and rectangular spatial integration;
- RK4 and velocity-Verlet primitives;
- informative wave and diffusion CFL limits.

These numerical operators do not define Agencity physics.

## Classical dynamics in 1.1.2

The research dynamics now exposed by `agencitylab.fields.dynamics` are:

```text
conservative KG: phi_tt = laplacian(phi) - g(phi)

dissipative KG:  phi_tt = laplacian(phi) - Gamma phi_t - g(phi), Gamma >= 0

TDGL:            phi_t = (laplacian(phi) - g(phi)) / Gamma, Gamma > 0
```

The conservative solver uses velocity-Verlet; the velocity-dependent dissipative equation is integrated as a first-order system with RK4; TDGL also uses RK4. See `classical_field_dynamics.md` for the numerical and boundary contracts.

## Coherent structures in 1.1.2

`agencitylab.fields.coherent` provides a real-sector domain-wall reference, two-dimensional U(1) vortex construction with caller-supplied radial profile, radial residual evaluation, and spatial winding diagnostics. The domain wall is explicitly a real-sector/Z2 reference and is not promoted to a generally stable defect of the full complex U(1) vacuum manifold. See `coherent_structures.md`.

## Data models and provenance

`DynamicalAgencityFieldState` and `DynamicalAgencityFieldSolution` are research-layer contracts for autonomous fields. `ScientificStatus` distinguishes `canonical`, `experimental`, `research`, and `speculative`; parameter provenance records whether values are user supplied, context supplied, benchmark conventions, source-document references, mathematically derived, or implementation conventions.

## Still outside 1.1.2

The following remain future work in the 1.1.x series:

- Agencity thermodynamics and temperature-dependent phase laws;
- gravity coupling;
- quantization and agentons;
- cosmological applications.

The canonical scalar engine remains unchanged.
