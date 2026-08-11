# Principles

This page summarizes the principle set stated in the accepted second-edition theory. These theoretical principles must not be confused with software test guarantees or with empirical validation.

## Principle 0 — Existence and objectivity

For an observable system with admissible $u(t)$ and a defined characteristic power $P_c(t)$, the theory treats $b(t)$ as an externally computable observable, defined almost everywhere without access to the internal mechanism. The scalar construction is translation-invariant under the corresponding normalization assumptions.

## Principle 1 — Agential power balance

The theory introduces a complex power-balance form

$$
b(t)=\Pi_{\mathrm{int}}(t)-\Phi_{\mathrm{diss}}(t)+S_{\mathrm{ext}}(t),
$$

separating internal production, dissipation, and external source terms. This is a theoretical balance statement; the v1.0 scalar reference API does not invent these components when they are not provided by a model.

## Principle 2 — Irreversibility

The causal construction is not invariant under time reversal. The theory associates this with an informational arrow of agency and with non-negative dissipation magnitude.

## Principle 3 — Contrast and critical surface

The surface

$$
\Sigma=\{D=S\}
$$

corresponds to $J=0$ and separates $D>S$ from $D<S$ regimes. Crossing this surface is a theory-facing transition diagnostic.

## Principle 4 — Orientation and coherence

Structural coherence is linked to the stability of

$$
\Theta(t)=\operatorname{atan2}(O(t),M(t)).
$$

The theory characterizes coherent organisation using structure $S>0$, low orientation variance, and significant $|b|$. Numerical meanings of “low” and “significant” remain contextual diagnostics rather than universal constants.

## Principle 5 — Minimality

The theory introduces an agential action proportional to the time integral of $|b|^2$ and identifies the critical condition $b=0$ / $D=S$ for $S>0$ with its passive reference/minimal condition.

## Principle 6 — Memory/coherence coupling

Temporal variations of memory $M$ and organisation $O$ contribute to the evolution of structural orientation and agencity magnitude. This motivates analysis of orientation stability without redefining the canonical state.

## Principle 7 — Extensivity and intensivity

The observable is decomposed as

$$
b=P_c\,\beta,
$$

with $P_c$ carrying the extensive physical scale and $\beta$ the intrinsic state. The theory discusses vector addition for independent-system agencities.

## Principle 8 — Structural stability

The theory associates changes around the critical contrast $J=0$ with possible transitions between fixed-point and oscillatory regimes. Specific bifurcation claims remain subject to their stated model assumptions and validation.

## Important v1.0 clarification

There is **no canonical saturation principle requiring $\beta\in(-1,1)$** in the accepted scalar formulation implemented by AgencityLab 1.0. For $S>0$, $|\beta|=|J|$, and the logarithmic contrast can grow without a universal unit bound. Historical documentation that described `beta` as a product of `tanh` factors is legacy material and is not the v1.0 theory mapping.
