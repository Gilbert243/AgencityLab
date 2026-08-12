---
orphan: true
---

# Classical Agencity Gravity

**Software version:** 1.1.2  
**Scientific status:** `research`

This module implements small numerical evaluators for the classical gravity coupling described in *Agencity — Advanced Mathematical Foundations and Extensions*, Volume 2, primarily Chapter 19 with the compact action of Chapter 23. It is a research implementation of the manuscript, not experimental validation of Agencity gravity and not a general-relativity solver.

The canonical observable pipeline `u -> beta -> b` is unchanged.

## Metric convention

The Gravity package uses the convention stated explicitly in Chapter 19:

```text
signature (-, +, +, +)
```

This is **not** the convention used by the flat classical field dynamics derived in Chapter 16 and implemented in `agencitylab.fields.dynamics`, where the manuscript writes the component Klein-Gordon equation with signature:

```text
signature (+, -, -, -)
```

AgencityLab does not silently identify these conventions.

There is an additional source-level sign issue worth preserving explicitly. Chapter 16 writes the flat equation in the `(+,-,-,-)` convention as

```text
phi_tt - laplacian(phi) + potential.gradient(phi) = 0.
```

Chapter 19 switches to `(-,+,+,+)` but states its curved equation in the formal form

```text
box(phi) + potential.gradient(phi) - xi R phi = 0.
```

For `(-,+,+,+)`, flat `box(phi) = -phi_tt + laplacian(phi)`. Therefore a metric-sign swap alone does not turn the Chapter-19 equation into the Chapter-16 equation: the kinetic operator changes sign while the potential-gradient term in the printed source does not. This package implements Chapter 19 literally and tests this difference. It does **not** change `fields.dynamics` or silently rewrite either source equation.

## Geometry utilities

`minkowski_metric()` and `minkowski_inverse_metric()` expose the Gravity `(-,+,+,+)` convention explicitly.

`sqrt_minus_g(metric)` evaluates `sqrt(-det(g))` for supplied finite `(...,4,4)` metric arrays.

`metric_with_perturbation(background_metric, perturbation)` implements only the algebraic representation

```text
g_mu_nu = eta_mu_nu + h_mu_nu.
```

It is not a gravitational-wave solver and does not compute curvature.

## Matter and gravitational action

Chapter 19 defines the matter action integrand from the supplied inverse metric, field derivatives, invariant volume factor, and the existing quartic field potential. The implementation reuses `QuarticAgencityPotential`; Gravity contains no second implementation of the potential or its gradient.

The package exposes:

- `matter_lagrangian_density(...)` for the bracketed matter Lagrangian before `sqrt(-g)`;
- `matter_action_density(...)` for the invariant matter action integrand;
- `nonminimal_coupling_density(...)` for `-1/2 xi sqrt(-g) R |phi|^2`;
- `einstein_hilbert_density(...)` for `sqrt(-g) R / (16 pi G)`;
- `total_gravity_field_lagrangian_density(...)` for the sum of those action integrands.

All required geometric objects are supplied explicitly. No metric, curvature, or spacetime geometry is inferred from `phi`.

## Minimal and conformal coupling

Minimal coupling is explicitly named as:

```text
xi = 0
```

The named `conformal_coupling()` helper returns:

```text
xi = 1/6
```

only as the Chapter-19 conformal value for the massless four-dimensional Klein-Gordon equation. It is **not** a universal default.

## External U(1) gauge potential

Chapter 23 writes the compact derivative as

```text
D_mu phi = partial_mu phi - i A_mu phi
```

if a gauge field is introduced. `covariant_scalar_derivative(...)` supports an **external caller-supplied** `A_mu` using exactly this algebraic rule. If `A_mu` is omitted, a scalar has `nabla_mu phi = partial_mu phi`.

This package does not implement a Maxwell action, Yang-Mills theory, gauge-field stress tensor, gauge-field evolution, or an autonomous gauge solver.

## Minimal stress-energy tensor

For `xi = 0`, `stress_energy_tensor(...)` implements the complete tensor printed in Chapter 19, including the explicit symmetrisation of the complex derivative product. The result is mathematically real for a real metric; the implementation verifies that any imaginary component is consistent with floating-point roundoff before returning the real tensor.

### Intentional limitation for nonminimal coupling

The accepted Volume-2 text does **not** provide the complete nonminimal stress-energy formula. It states only that for `xi != 0` additional terms occur involving the Einstein tensor and second derivatives of `|phi|^2`.

Accordingly:

- the nonminimal **action** is supported;
- the nonminimal **curved field equation** is supported;
- `stress_energy_tensor(..., xi != 0)` raises `NotImplementedError`.

No external GR formula is imported and presented as if it were specified by the Theory of Agencity.

## Einstein equation residual

`einstein_equation_residual(...)` evaluates

```text
G_mu_nu - 8 pi G T_mu_nu.
```

The Einstein tensor and stress-energy tensor must both be supplied explicitly with compatible `(...,4,4)` shapes, and physical `G` must be finite and strictly positive.

The function does not solve for the metric and is not a 3+1 numerical-relativity implementation.

## Curved field residual

`curved_field_residual(...)` evaluates the Chapter-19 equation as

```text
box_phi + potential.gradient(phi) - xi R phi.
```

For the existing quartic potential, `potential.gradient(phi)` is the mathematically equivalent nonsingular representation of the source term printed as `V'(|phi|) phi / |phi|`. In particular, `phi = 0` is handled directly without adding an epsilon.

`minkowski_box(phi_tt, spatial_laplacian)` makes the Gravity signature explicit:

```text
box_phi = -phi_tt + spatial_laplacian
```

in natural units.

## U(1) invariance

The tests cover global phase rotations

```text
phi -> phi exp(i alpha)
```

with derivatives transformed consistently. The modulus, shared potential, ungauged matter Lagrangian, and minimal scalar-field stress-energy tensor remain invariant as expected.

## Validation and numerical scope

The numerical API validates finite fields and geometry, explicit four-component derivative axes, `(...,4,4)` tensor shapes, finite `xi`, positive finite `G`, and compatible external gauge-field shapes. General NumPy broadcasting between unrelated field and geometry shapes is intentionally rejected.

A small roundoff tolerance is used only to verify quantities that are mathematically real. It is not inserted into a physical denominator or field equation.

## What is not implemented

This package intentionally does not provide:

- a generic Christoffel/Riemann/Ricci symbolic engine;
- a 3+1 Einstein solver;
- automatic metric evolution;
- a gravitational-wave evolution equation;
- FLRW or scale-factor evolution;
- inflation or dark-energy solvers;
- thermodynamics;
- quantum fields, Fock states, or agentons;
- autonomous gauge dynamics.

The manuscript's qualitative linearised-gravity and cosmological discussions therefore remain theory context, not operational solvers in AgencityLab 1.1.2.

## Public API scope

The direct research API is available only from:

```python
import agencitylab.gravity
```

This branch does not add top-level `agencitylab` exports. Public integration and versioning are deferred to the planned 1.1.3 integration work.
