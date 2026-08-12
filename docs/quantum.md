---
orphan: true
---

# Quantum Agencity primitives

**Scientific status:** `speculative`

This module implements a deliberately small computational subset of *Agencity — Advanced Mathematical Foundations and Extensions*, Volume 2, Chapter 21. It quantises the autonomous field `phi` proposed in the field-theory extension. It does **not** quantise the canonical observable pipeline `u -> beta -> b`, and it does not constitute experimental evidence for agentons or quantum Agencity.

## Implemented source relations

For the broken-symmetry phase `lambda > 0`, the shared quartic field potential has vacuum amplitude

`v = sqrt(lambda / mu)`.

The Chapter-21 excitations are represented by the stated mode relations

`m_h^2 = 2 lambda`,

`omega_h(k) = sqrt(k^2 + m_h^2)`,

and a massless Goldstone mode

`m_theta^2 = 0`, `omega_theta(k) = |k|`.

The momentum-space propagator evaluators implement

`D_h(k) = i / (k^2 - m_h^2 + i epsilon)`

and

`D_theta(k) = i / (k^2 + i epsilon)`.

`epsilon` is always supplied explicitly by the caller as a numerical regulator. AgencityLab does not insert a machine epsilon into these denominators as a physical prescription.

The one-loop helper returns only the coefficient written in the source,

`beta(mu) = 5 mu^2 / (16 pi^2) + O(mu^3)`.

The `O(mu^3)` terms are not guessed or modelled.

## Finite Fock-space approximation

Chapter 21 states the infinite-dimensional bosonic algebra. `agencitylab.quantum.fock` supplies a finite number-basis approximation for numerical experiments. A finite matrix cannot obey `[a, a^dagger] = I` on every retained state. For a basis of size `N`, the implementation exposes the exact truncation defect

`[a, a^dagger] - I = -N |N-1><N-1|`.

This is a numerical approximation, not a modification of the source commutation relation.

## Agencity uncertainty relation

For constant bridge parameters, Chapter 21 combines `b = sqrt(P_c / tau) phi` with the field uncertainty relation to state

`Delta b Delta b_dot >= (hbar / 2) P_c / tau`.

`agencity_uncertainty_lower_bound` requires explicit scalar `P_c`, `tau`, and `hbar`. `P_c = 0` is valid and gives an exact zero bound. The helper intentionally does not claim this formula for time-dependent bridge parameters, because differentiating a time-dependent prefactor introduces additional terms.

## Deliberately not implemented

The source does not provide enough operational detail here to justify a generic path-integral engine, vacuum-energy regularisation, renormalisation scheme, scattering engine, lattice QFT framework, autonomous quantum-gravity dynamics, or numerical vertex coefficients beyond the explicit relations above. Those are not inferred from external QFT conventions and presented as Agencity theory.
