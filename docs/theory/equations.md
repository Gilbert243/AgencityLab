# Equations

## Normalization

$$
u^*(\xi) = \frac{u(\xi)}{u_{\mathrm{ref}}}
$$

## Activation

$$
X^*(\xi) = \frac{d u^*}{d \xi}
$$

## Activity

$$
A^*(\xi^*) = X^*(\xi^* + 1/2) - X^*(\xi^* - 1/2)
$$

## Memory and organization

$$
M(\xi^*) = \tanh(\mathrm{crm}_\tau(A^*)(\xi^*)), \qquad
O(\xi^*) = \tanh(\mathrm{crm}_\tau(X^*)(\xi^*))
$$

## Agencement

$$
\beta(\xi^*) = \tanh\!\big(X^*(\xi^*)(1 + A^*(\xi^*))\big)\,
\tanh\!\big(M(\xi^*) + O(\xi^*)\big)
$$

## Observable Agencity

$$
b(\xi) = P_c(\xi)\,\frac{\beta(\xi^*) - \beta(\xi^* - \Delta^*)}{\Delta^*}
$$
