# Mathematical extension completeness

This page implements source-defined mathematical material from Volume 2 that is
not part of the canonical scalar `u -> beta -> b` engine. The stable canonical
formula is not replaced by any historical candidate or analysis helper here.

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
algorithm printed in Chapter 13 and Appendix C. Evaluators for `Phi1` and
`Phi3` do not replace that algorithm.

For `Phi3`, Volume 2 says that `p_k` are frequencies of discretised angle bins
but does not prescribe a universal number or placement of bins. Therefore
`orientational_entropy_criterion()` requires explicit `bin_edges` from the
caller. No universal discretisation is invented. An explicit `valid_mask`
may exclude samples at which structural orientation is undefined, such as
`S = 0`.

These criteria are selection/analysis extensions. They do not silently infer
a physically supplied characteristic time `tau`, and a signal-derived optimum
is not promoted into a physical parameter without an explicit user decision.

## Chapter 14 historical intensity candidates

Chapter 14 records formulas that were tested **before** the canonical
logarithmic contrast was retained. AgencityLab exposes the unambiguous printed
candidates under `agencitylab.extensions` for comparison and historical
reproducibility only:

```text
I1 = |X| + |A X| + |M| + |O|
J1 = ln(I1)

I2_raw = (|X| + |A X|) / (|M| + |O|)
```

`sum_intensity()`, `sum_log_intensity()`, and `raw_ratio_intensity()` preserve
their source singularities. Exact rest gives `J1 = -inf`; a zero denominator
in the raw ratio is not hidden with machine epsilon.

The source then prints the offset expression

```text
e + (|X| + |A X|) / (e + |M| + |O|)
```

at the end of Section 14.3 and prints the same algebraic expression again as
`I3` in Section 14.4. Because those two displayed formulas are identical,
`printed_offset_ratio_candidate()` implements the common printed expression
once with the fixed `e = exp(1)`. The library does not invent a distinction
between the two source labels.

Scientific status: `experimental` historical/reference formulas. None is a
selectable replacement for canonical `J` inside `compute_agencity()`.

## Chapters 4 and 10 robustness relations

The analysis layer exposes two exact relations used by the mathematical text:

```text
partial J / partial e = (S-D) / ((e+D)(e+S))

P_c = P_c0 (1 + epsilon)
=> b = b0 + epsilon b0
```

`logarithmic_contrast_offset_sensitivity()` evaluates the first relation at the
canonical fixed constant `e = exp(1)`; it does **not** make `e` tunable.
`multiplicative_power_perturbation()` evaluates the second relation without
assuming a stochastic distribution for `epsilon`.

These are mathematical/diagnostic consequences, not modifications of `J` or
`P_c`.

## Chapter 11 inverse problem: recoverable information only

Volume 2 proves that `u -> b` is non-injective. Consequently AgencityLab does
not provide an inverse reconstruction of `u`. With known strictly positive
`P_c`, however, the source states that

```text
beta = b / P_c
|J| = |beta|
```

and that the direction of `(M,O)` can be recovered only up to a sign from the
phase. `recoverable_agencity_signature()` therefore returns `beta`, `|J|`, and
an orientation modulo `pi`. The direction is reported undefined when
`beta = 0`, because `b` alone cannot then distinguish a structural zero from a
critical-surface zero `J = 0`.

The helper refuses `P_c = 0`: the forward equation gives `b = 0` for every
intrinsic state at zero characteristic power, so division would fabricate
information that is not present in the observable flux.

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

The Chapter-14 offset candidate collision described above is handled the same
way: the common printed expression is reproducible, but no undocumented
mathematical difference is manufactured between the two labels.

Likewise, no closed autonomous equation for the canonical flux `b` is added:
Volume 2 Section 23.6 explicitly leaves `b_eq` undetermined as future work.
