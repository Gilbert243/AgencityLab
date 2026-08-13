# Effective beta field and flat-field conservation laws

**Scientific status:** `research`.

This page documents source-defined field material from Volume 2 that is
separate from the canonical observable pipeline.  Nothing here changes
`u -> beta -> b`, CRM, `P_c`, `tau`, `w`, or the exact `S = 0 => beta = 0`
convention.

## Three different objects must remain distinct

AgencityLab now represents three source layers explicitly:

1. `beta_obs(x,t)` is computed from an observable `u(x,t)` by the local
   temporal canonical pipeline.  Spatial orchestration is experimental; the
   scalar computations are canonical.
2. Chapter 15 then **postulates** an effective dynamical equation directly for
   a complex field called `beta`.  AgencityLab exposes this separately under
   `agencitylab.fields.effective_beta`; it is a research model and is not the
   algorithm that computes `beta_obs`.
3. Chapters 15--16 introduce the canonically normalised autonomous field
   `phi = sqrt(P_c tau) beta`, after which the Klein--Gordon field theory is
   written in terms of `phi`.

## Chapter 15 effective beta equation

Volume 2 Eqs. (15.2)--(15.3) state

```text
partial_t beta + v . grad(beta)
    = D_diff laplacian(beta) + a beta - b_sat |beta|^2 beta
```

where the source symbols `D` and `b` are respectively a positive diffusion
coefficient and a positive nonlinear saturation coefficient.  Those symbols
collide with the canonical dynamic intensity `D` and Agencity flux `b`.
AgencityLab therefore uses the explicit argument names
`diffusion_coefficient` and `saturation_coefficient`.

The direct complex-field equation is implemented by `effective_beta_rhs()`.
The homogeneous non-zero stationary amplitude for `a > 0` is
`sqrt(a / b_sat)` and is exposed explicitly without selecting a phase.

### Printed amplitude/phase equation

The source prints Eqs. (15.4)--(15.5) after writing `beta = R exp(i Theta)`.
Eq. (15.4) follows directly from the real part of Eq. (15.2).  For the phase,
a direct decomposition of Eq. (15.2) gives, for `R > 0`,

```text
R (partial_t Theta + v . grad Theta)
  = D_diff [2 grad R . grad Theta + R laplacian Theta]
  = (D_diff / R) div(R^2 grad Theta).
```

The printed Eq. (15.5), however, places `D_diff div(R^2 grad Theta)` on the
right-hand side without the corresponding `1/R` factor (or, equivalently,
without an additional factor of `R` on the left).  AgencityLab does **not**
silently repair this source-level mismatch.  The unambiguous complex equation
(15.2)--(15.3) is implemented; an independent Eq. (15.5) evaluator is deferred
until the source convention is explicitly resolved by the maintainer.

## Chapter 16 flat-field convention

Chapter 16 uses Minkowski signature

```text
(+,-,-,-)
```

and the action density

```text
L = 1/2 partial_mu phi partial^mu conjugate(phi) - V(|phi|).
```

This is deliberately separate from the Chapter-19 Gravity package, whose
source convention is `(-,+,+,+)`.

`flat_field_lagrangian_density()` evaluates the Chapter-16 density from
caller-supplied coordinate derivatives.  `flat_energy_momentum_tensor()`
implements Eq. (16.4) and `u1_noether_current()` implements the global-U(1)
current with the chapter-internal sign fixed by Eq. (16.7):

```text
J^mu = R^2 partial^mu Theta.
```

`phase_noether_current()` exposes that amplitude/phase form directly.
`radial_equation_residual()` evaluates Eq. (16.6) from caller-supplied
`box(R)` and `(partial Theta)^2`; it does not invent a spacetime discretisation.

Conservation, `partial_mu T^{mu nu} = 0` and `partial_mu J^mu = 0`, is an
**on-shell mathematical statement**.  The software returns the tensors and
currents; it does not project arbitrary numerical data onto a conserved state.

## Appendix B beta-field formulation

Appendix B also records a separate field-theoretic formulary written directly
in terms of `beta`:

```text
L_beta = 1/2 P_c^2 partial_mu beta partial^mu conjugate(beta) - V(|beta|)
box(beta) + potential_gradient(beta) / P_c^2 = 0
J_beta^mu = P_c^2 J_unscaled^mu
```

with the corresponding `P_c^2` energy--momentum tensor.

These equations are implemented under explicitly prefixed
`appendix_b_beta_*` names.  They are **not silently identified** with the
Chapter-16 `phi` formulation.  In particular, direct substitution of the
constant bridge `phi = sqrt(P_c tau) beta` into the Chapter-16 kinetic term
would produce a different coefficient unless additional relations are
introduced; the accepted source does not supply such a reconciliation.

For this Appendix-B research formulation, `P_c` is therefore required to be a
finite positive scalar when the equation divides by `P_c^2`.  This local field
contract does not alter the canonical observable rule: `P_c = 0` remains valid
for `b = P_c beta` and gives `b = 0` exactly.

## Deliberate non-implementations

This layer does not implement a closed autonomous equation for the canonical
flux `b`.  Volume 2 Section 23.6 explicitly leaves `b_eq` undetermined and
classifies that closure as future research.  It is therefore not a current
Agencity equation to be fabricated in software.
