# Canonical equations

This page records the scalar equations implemented by the v1.0 reference pipeline. Research extensions are documented separately and do not replace these definitions.

## Normalization and reduced coordinate

For an observable `u` and a physical/contextual reference amplitude `A_ref > 0`:

$$
u^*(t) = \frac{u(t)}{A_{\mathrm{ref}}}.
$$

With characteristic structural time $\tau > 0$:

$$
t^* = \frac{t}{\tau}.
$$

## Activation and activity

The reduced activation and activity are

$$
X^*(t^*) = \frac{d u^*}{d t^*},
\qquad
A^*(t^*) = \frac{d X^*}{d t^*}.
$$

The implementation uses finite differences as a numerical approximation of these continuous derivatives.

## Causal moving correlation, memory, and organisation

Let $\operatorname{CRM}_w$ denote the causal moving correlation using adjacent causal windows of width $w>0$.

$$
M(t) = \operatorname{CRM}_w[u^*](t),
$$

$$
O(t) = \operatorname{CRM}_w[u^*, X^*](t).
$$

The CRM convention returns zero when an empirical Pearson denominator is exactly zero. Numerical epsilon is not inserted into the correlation coefficient.

The characteristic time $\tau$ and the CRM width $w$ are distinct quantities. If the public API omits `w`, AgencityLab uses the software convention $w=\tau$; this is not a universal theoretical identity.

## Dynamic and structural intensities

$$
D(t) = \sqrt{(X^*)^2 + (A^* X^*)^2},
$$

$$
S(t) = \sqrt{M(t)^2 + O(t)^2}.
$$

## Logarithmic contrast

With the fixed theoretical constant $e=\exp(1)$:

$$
J(t) = \ln\!\left(\frac{e + D(t)}{e + S(t)}\right).
$$

Therefore $D=S$ implies $J=0$ exactly.

## Structural orientation

$$
\Theta(t) = \operatorname{atan2}(O(t), M(t)).
$$

For $S(t)>0$:

$$
U(t) = \frac{M(t)+iO(t)}{S(t)} = e^{i\Theta(t)},
$$

so $|U(t)|=1$.

When $S(t)=0$, the structural direction is undefined. The canonical software representation uses $U(t)=0$ together with the theory convention $\beta(t)=0$.

## Intrinsic agencity state

$$
\beta(t)=
\begin{cases}
J(t)U(t), & S(t)>0,\\
0, & S(t)=0.
\end{cases}
$$

For $S>0$, $|\beta|=|J|$. The state is therefore not universally bounded to $(-1,1)$; in the large-dynamic-intensity limit its magnitude has logarithmic growth through $J$.

## Observable agencity flux

For a positive physical/contextual characteristic power $P_c(t)$:

$$
\boxed{b(t)=P_c(t)\,\beta(t)}.
$$

This multiplicative relation is the stable v1.0 observable definition. The historical derivative-of-$\beta$ formula and historical `tanh` construction are not part of the accepted reference pipeline.

## Exact rest convention

For an exactly constant sampled observable, the reference implementation applies the accepted null-state convention:

$$
X^*=A^*=M=O=D=S=J=0,
\qquad U=\beta=b=0.
$$

The check is exact. Near-constant but non-constant signals are not forced to zero by a universal epsilon threshold.
