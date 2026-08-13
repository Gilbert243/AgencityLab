# Classical Dynamical Agencity Field

**Software version:** 1.1.2  
**Scientific status:** `research`

AgencityLab 1.1.2 exposes deterministic NumPy reference implementations of the classical autonomous-field dynamics described in Volume 2. These equations are research extensions of the Theory of Agencity; they are not part of the canonical observable pipeline and their numerical implementation is not empirical validation.

The canonical observable construction remains

```text
u -> beta -> b
```

and the autonomous field is reached only through the explicit research bridge

```text
phi = sqrt(P_c * tau) * beta.
```

## Shared quartic potential

All dynamics reuse `QuarticAgencityPotential` from the field-physics layer:

```text
V(phi) = -(lambda/2) |phi|^2 + (mu/4) |phi|^4

g(phi) = -lambda phi + mu |phi|^2 phi
```

No solver duplicates or modifies this potential.

## Conservative Klein-Gordon

In the dimensionless/natural-unit convention with `c = 1`, the implemented equation is

```text
phi_tt = laplacian(phi) - g(phi).
```

`klein_gordon_acceleration()` evaluates the right-hand side using the shared numerical Laplacian and potential. `simulate_klein_gordon()` uses the existing velocity-Verlet integrator and returns `DynamicalAgencityFieldSolution`.

## Dissipative Klein-Gordon

The dissipative research equation is

```text
phi_tt = laplacian(phi) - Gamma phi_t - g(phi),   Gamma >= 0.
```

`Gamma = 0` reduces exactly to the conservative acceleration primitive. `simulate_dissipative_klein_gordon()` integrates the first-order `(phi, phi_dot)` system with the existing RK4 primitive because the acceleration depends on velocity.

## Overdamped TDGL

The overdamped equation is

```text
Gamma phi_t = laplacian(phi) - g(phi),   Gamma > 0.
```

`simulate_tdgl()` uses RK4. No epsilon is inserted into `Gamma`; zero is rejected for this equation.

## Boundary handling

The dynamics reuse `PeriodicBoundary`, `DirichletBoundary`, and `NeumannBoundary` from the numerical infrastructure.

- Periodic boundaries are enforced through the spatial stencil.
- Dirichlet boundaries additionally require explicit stage/state projection in the time-integration layer so fixed field values remain fixed; `phi_dot` is zeroed on fixed faces for second-order dynamics.
- Neumann conditions remain stencil-enforced. The solver metadata explicitly states that no exact temporal state projection is claimed.

## Numerical status

The simulators use fixed time steps and are deterministic. Energy conservation/decrease checks in the test suite are numerical diagnostics of the implemented equations, not additional physical laws or empirical evidence.

No universal physical values are assigned to `lambda`, `mu`, `Gamma`, `dt`, or grid scales. Parameter provenance remains explicit.
