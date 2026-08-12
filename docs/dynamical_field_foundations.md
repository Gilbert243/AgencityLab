# Dynamical Agencity Field Foundations

**Software version:** 1.1.1  
**Scientific status:** `research` for autonomous `phi` physics; `experimental` for generic field numerics.

AgencityLab 1.1.1 integrates the common foundations needed for later autonomous field dynamics without yet claiming a Klein–Gordon, dissipative Klein–Gordon, or TDGL simulation API.

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

## Data models and provenance

`DynamicalAgencityFieldState` and `DynamicalAgencityFieldSolution` are research-layer contracts for autonomous fields. `ScientificStatus` distinguishes `canonical`, `experimental`, `research`, and `speculative`; parameter provenance records whether values are user supplied, context supplied, benchmark conventions, source-document references, mathematically derived, or implementation conventions.

## What 1.1.1 does not yet implement

The following remain future work in the 1.1.x series:

- physical Klein–Gordon time evolution;
- dissipative Klein–Gordon evolution;
- time-dependent Ginzburg–Landau evolution;
- domain walls, vortices, and phase-transition simulation helpers;
- Agencity thermodynamics;
- gravity coupling;
- quantization and agentons;
- cosmology.

The canonical scalar engine remains unchanged.
