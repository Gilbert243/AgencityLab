# Mathematical extension completeness

This page implements the source-defined parts of Volume 2 Chapters 12, 13 and
17 that are not part of the canonical scalar `u -> beta -> b` engine.

## Riemannian extension — Definition 12.4

Volume 2 defines, for a `C^2` curve on a Riemannian manifold `(M,g)`,

```text
X = gamma_dot
A = nabla_X X
D = sqrt(g(X,X) + g(A,X)^2).
```

It also notes that scalar quantities such as `||X||_g` or `g(A,X)` may be used
as inputs to the real-valued CRM operator. The same section explicitly says
that the detailed analysis is deferred.

AgencityLab therefore implements only the intrinsic kinematic primitives:

- `riemannian_inner_product`;
- `riemannian_speed`;
- `riemannian_dynamic_intensity`.

The caller supplies tangent vectors, covariant acceleration and a symmetric
positive-definite metric. The software does **not** invent a coordinate-chart
framework, a Levi-Civita connection solver, or a complete Riemannian CRM/state
pipeline that the source does not define operationally.

Scientific status: `experimental` mathematical extension.

## Chapter 13 window criteria

The source defines three objective functionals of a candidate CRM width `w`:

```text
Phi1(w) = (1/T) integral |J_w(t)| dt          maximise
Phi2(w) = mean sliding variance of Theta_w    minimise
Phi3(w) = - sum_k p_k ln(p_k)                 maximise
```

The existing automatic window selector continues to use `Phi2`, matching the
algorithm printed in Chapter 13 and Appendix C. This change adds evaluators
for `Phi1` and `Phi3`; it does not replace that algorithm.

For `Phi3`, Volume 2 says that `p_k` are frequencies of discretised angle bins
but does not prescribe a universal number or placement of bins. Therefore
`orientational_entropy_criterion()` requires explicit `bin_edges` from the
caller. No universal discretisation is invented. An explicit `valid_mask`
may exclude samples at which structural orientation is undefined, such as
`S = 0`.

These criteria are selection/analysis extensions. They do not silently infer
a physically supplied characteristic time `tau`, and a signal-derived optimum
is not promoted into a physical parameter without an explicit user decision.

## Chapter 17 dimensionless coherent-field formulation

For the broken phase `lambda > 0`, `mu > 0`, Volume 2 introduces

```text
phi = sqrt(lambda/mu) psi
xi = 1/sqrt(lambda)
-laplacian(psi) - psi + |psi|^2 psi = 0
W(psi) = (1/4)(1-|psi|^2)^2.
```

AgencityLab exposes this rescaling and static residual under
`agencitylab.fields.coherent`. The spatial Laplacian remains the shared
Numerics operator. This is not a second implementation of the physical
quartic `phi` potential.

Scientific status: `research`, consistent with the autonomous field/coherent
structure layer.

## Source formulas deliberately not guessed

The cycle-area formula printed in Chapter 8 is extracted as

```text
A = 1/2 Im integral beta(t) beta_dot(t) dt.
```

Read literally for a closed differentiable curve, that integrand is a total
derivative and the integral vanishes. The surrounding text calls it the
*enclosed algebraic area*, which ordinarily requires a complex conjugation or
an equivalent real-plane cross product. Because the accepted source as
currently rendered does not resolve that typography unambiguously,
AgencityLab does not silently choose one formula.

Likewise, no closed autonomous equation for the canonical flux `b` is added:
Volume 2 Section 23.6 explicitly leaves `b_eq` undetermined as future work.
