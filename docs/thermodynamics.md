:orphan:

# Unified Agencity Thermodynamics

**Scientific status: `research`.** This layer implements mathematical evaluators from the accepted theory documents. It is neither part of the canonical `u -> beta -> b` definition nor evidence of empirical validation.

## Source hierarchy and scope

The primary source is *Agencity — Advanced Mathematical Foundations and Extensions*, Volume 2, Chapter 18. Volume 1 is used only where an explicitly distinct formula belongs to it, notably Appendix H Eq. (H.26) for contrast agencial entropy and the conditional Landauer relations H.23–H.25.

This module reuses the existing field Physics, Numerics, Dynamics, and shared field-result contracts. It does not duplicate the quartic potential, its gradient, Klein–Gordon equations, TDGL, spatial operators, or time integrators.

## Source formula versus numerical evaluator

### Dissipation and local entropy production

Volume 2 Eq. (18.3) states

```text
partial_t H + div(J_E) = -Gamma |partial_t phi|^2
```

and Eq. (18.4) gives

```text
Q_dot = Gamma |partial_t phi|^2
sigma = Q_dot / T_eff
```

The numerical evaluators are `dissipation_density`, `entropy_production_density`, `total_dissipated_power`, and `total_entropy_production`.

`Gamma` is finite and non-negative in these evaluators. `Gamma = 0` gives exact zero dissipation and entropy production. This does **not** alter the Dynamics contract: existing TDGL still requires `Gamma > 0`.

`T_eff` is supplied explicitly and must be finite and strictly positive. No epsilon is inserted into a temperature denominator and no signal-derived temperature estimate is part of the research contract.

### Energy balance

Chapter 18 names the energy flux `J_E` but does not provide a unique numerical spatial discretisation for it in this implementation context. Therefore the library does **not** invent an energy-flux primitive. `energy_balance_residual(dH_dt, div_j_e, phi_dot, gamma)` evaluates

```text
dH_dt + div_j_e + Gamma |phi_dot|^2
```

and zero is the source-equation condition.

### Temperature-dependent coefficient

Section 18.4 gives exactly

```text
lambda(T) = a (T_c - T)
```

implemented by `temperature_dependent_lambda`. `a`, `T_c`, and `T` are explicit inputs. No sign of `a` is imposed silently. If `a > 0`, then the mathematical consequences are `T < T_c -> lambda > 0`, `T = T_c -> lambda = 0`, and `T > T_c -> lambda < 0`.

## Two distinct agencial entropies

These quantities must not be merged.

### Field agencial entropy — Volume 2

Volume 2 Eq. (18.5) defines

```text
S_ag_field = (a / 2) integral |phi|^2 dV
```

implemented as `field_agencial_entropy(phi, a, grid)`. It supports real and complex fields and uses the existing spatial quadrature. The source writes `S_ag >= 0`; that sign follows under an appropriate positive-`a` physical context, so the software does not invent a hidden sign constraint on `a`.

### Contrast agencial entropy — Volume 1

Volume 1 Appendix H Eq. (H.26) defines a different quantity:

```text
S_ag_contrast = -k_B ln(1 - |J| / J_max)
```

implemented as `contrast_agencial_entropy(J, j_max, k_b)`. The finite real-valued logarithm requires `J_max > 0` and `|J| < J_max`. The manuscript gives, as an example scale, `J_max = ln(1 + D_max/e)`.

The historical `agential_entropy` Shannon-style array helper corresponds to neither formula. It is retained only as a deprecated compatibility placeholder and is not an alias for either accepted entropy.

## Second principle

Volume 2 Eq. (18.6) states

```text
d/dt (S_ag + S_therm) = integral sigma dV >= 0
```

`second_law_residual(dS_ag_dt, dS_therm_dt, total_sigma)` evaluates the left side minus the right side. It never forces the result to zero and never clips a negative diagnostic.

## Modulus Law

Volume 2 Eq. (18.7) states

```text
|b| >= P_diss + T_amb Sdot_int
```

`modulus_law_margin` evaluates

```text
abs(b) - (P_diss + T_amb * Sdot_int)
```

and `modulus_law_satisfied` tests whether the resulting margin is non-negative. The evaluator accepts canonical `b` or an external supplied value. It never modifies `b` and never clips `P_diss + T_amb Sdot_int`. Positive and negative `Sdot_int` therefore remain visible.

## Phase Law and the symbol `O`

Volume 2 Eq. (18.8) uses the symbol `O` for the imaginary field component

```text
O = |phi| sin(Theta) = Im(phi)
```

but canonical Agencity already uses

```text
O = CRM[u, u_dot]
```

for **organisation**. The thermodynamics API deliberately avoids an argument named `O`. It uses `phase_component`, `phi_imaginary_component`, or `imag_phi` terminology.

The empirical Phase-Law relation is evaluated as

```text
phase_component ~= alpha log10(P_diss / (T_amb |Sdot_int|)) + beta_fit
```

The logarithmic ratio must be finite and strictly positive. `Sdot_int = 0` is undefined and raises a clear error; no epsilon is added.

`PhaseLawFit` accepts explicit user-supplied coefficients. `thermal_reference_phase_fit()` is a **named empirical reference** containing the values reported for the manuscript's thermal systems:

```text
alpha ~= 0.82
beta_fit ~= -1.50
R^2 ~= 0.87
```

These values are not universal constants and are never silent defaults.

## Conditional Landauer relations

Volume 1 Appendix H explicitly gives, in the version constructed with exact Landauer equality,

```text
P_c = k_B T_eff / tau                 (H.23)
I_dot_struct = |beta| / tau           (H.24)
|b| = k_B T_eff I_dot_struct          (H.25)
```

The corresponding helpers are implemented in `thermodynamics.landauer` and documented as **conditional relations**, not replacements for canonical `b = P_c beta`. No generic `k_B T ln 2` rule is promoted to an Agencity law.

## Historical placeholders

The previous thermodynamics package contained generic helpers such as a clipped energy difference, `energy/dof` temperature estimate, monotonic-series second-law check, and Shannon-style `agential_entropy`. These are not authoritative theory formulas. Where retained for compatibility, they emit deprecation warnings and their docstrings state that they are legacy heuristics.

## Explicit exclusions

This thermodynamics layer does not implement gravity, Einstein equations, curved spacetime, quantum fields, agentons, FLRW, inflation, dark energy, a cosmological solver, or a closed autonomous equation for `b`. In particular, no `tau_b db/dt = ...` closure is introduced.

The Modulus Law and Phase Law are **evaluated**, never imposed on the canonical pipeline.
