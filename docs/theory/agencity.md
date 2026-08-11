# Theory of Agencity

Agencity is introduced as a complex-valued observable intended to characterize the structuring dynamics of an observable system without requiring access to its internal mechanism.

The accepted scalar construction follows the canonical pipeline:

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,Theta -> beta -> b
```

## Intrinsic state

The dynamic and structural intensities are

$$
D(t)=\sqrt{(X^*)^2+(A^*X^*)^2},
\qquad
S(t)=\sqrt{M(t)^2+O(t)^2}.
$$

The logarithmic contrast is

$$
J(t)=\ln\!\left(\frac{e+D(t)}{e+S(t)}\right).
$$

When $S(t)>0$, the structural direction is

$$
U(t)=\frac{M(t)+iO(t)}{S(t)}=e^{i\Theta(t)},
\qquad
\Theta(t)=\operatorname{atan2}(O(t),M(t)).
$$

The intrinsic agencity state is

$$
\beta(t)=J(t)U(t) \quad \text{for } S(t)>0.
$$

When $S(t)=0$, the accepted convention is

$$
\beta(t)=0.
$$

## Observable agencity flux

The observable flux is

$$
\boxed{b(t)=P_c(t)\,\beta(t)},
$$

where $P_c$ is the physical/contextual characteristic power.

The historical derivative-of-$\beta$ expression is not the canonical observable used by the v1.0 reference implementation.

## Memory width and characteristic time

The characteristic structural time $\tau>0$ and CRM window width $w>0$ are separate quantities. Volume 2 explicitly keeps the distinction, while often studying the convenient case $w=\tau$.

AgencityLab therefore preserves an explicitly supplied `w`. When `w` is omitted, the stable API uses `w=tau` as an implementation convention rather than a universal identity.

## Interpretation boundary

`beta` is the intrinsic complex state built from logarithmic contrast and structural orientation; `b` adds the characteristic-power scale. Neither `beta != 0` nor large dynamic intensity alone defines coherent or real agencity.

Coherence and real-agencity assessment belong to a separate diagnostic layer that can consider structural validity, angular stability, and contextual significance of `|b|` without modifying the canonical equations.

## Scientific status

Version 1.0 provides a stable reference implementation and a deterministic validation laboratory. The software release does not by itself establish universal empirical validity of the theory or of its research extensions.
