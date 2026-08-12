# Cosmological Agencity

**Scientific status:** `speculative`

This application layer implements the homogeneous flat-FLRW equations stated in *Agencity — Advanced Mathematical Foundations and Extensions*, Volume 2, Chapters 20 and 22. It treats the autonomous field `phi` as a hypothetical cosmological matter source. It is not an observationally validated cosmological model and does not modify the canonical observable construction `u -> beta -> b`.

## Homogeneous field and stress-energy

For a spatially homogeneous field,

`rho_phi = 1/2 |phi_dot|^2 + V(|phi|)`

and

`p_phi = 1/2 |phi_dot|^2 - V(|phi|)`.

AgencityLab reuses the shared `QuarticAgencityPotential` and shared field-energy primitives. It does not copy the quartic potential into the cosmology package.

The equation-of-state helper evaluates

`w_phi = p_phi / rho_phi`

only where `rho_phi != 0`. No EPS is inserted at zero density.

## Flat FLRW equations

The implemented source equations are

`phi_ddot + 3 H phi_dot + V'(|phi|) phi/|phi| = 0`,

`H^2 = (8 pi G / 3) rho_phi`,

and

`H_dot + H^2 = -(4 pi G / 3) (rho_phi + 3 p_phi)`.

The shared `potential.gradient(phi)` supplies the field-force term, including the exact regular value at `phi = 0`; no `phi/(|phi| + EPS)` convention is introduced.

`simulate_flat_flrw` requires the caller to select an initial `expanding` or `contracting` branch. The initial Hubble value is obtained from the first Friedmann equation. The solver then evolves `phi`, `phi_dot`, `a`, and `H` using the existing generic RK4 integrator and the acceleration equation. The first Friedmann equation is retained as a numerical residual rather than projected after every step. Numerical drift is therefore observable instead of silently corrected.

## Quartic-vacuum limitation

For the minimal broken-symmetry quartic potential with `lambda > 0`, the vacuum energy is

`V_min = -lambda^2 / (4 mu) < 0`.

The accepted source explicitly notes that this negative minimum cannot by itself account for a positive observed dark-energy density. AgencityLab preserves that result. `initial_hubble_from_friedmann` rejects negative `rho` because no real flat-FLRW Hubble value can satisfy the first Friedmann equation with that sole source.

No cosmological constant, positive offset, quintessence term, or modified potential is silently added.

## Phase transition and inflation

Chapter 22 discusses `lambda(T) = a (T_c - T)` and possible cosmological phase transitions. The existing thermodynamics primitive already evaluates this relation. This package does not invent a temperature-versus-scale-factor law or a thermal solver.

The source further states that inflation would require a sufficiently flat plateau and therefore a modification of the simple quartic potential, for example an added constant term. Such a modification is not implemented here because it is not part of the shared reference potential contract.

## Deliberately not implemented

This layer does not provide cosmological perturbation theory, CMB predictions, inflationary spectra, dark-energy fitting, Kibble-Zurek defect production, quantum vacuum stability, a generic Einstein solver, or observational parameter inference. Those require additional theory and modelling choices beyond the explicit homogeneous equations implemented here.
