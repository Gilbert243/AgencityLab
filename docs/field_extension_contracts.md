# Field extension contracts

AgencityLab distinguishes observable Agencity fields from autonomous dynamical fields and from later thermodynamic, gravitational, quantum, and cosmological extensions. This document defines shared scientific-status, data-model, units, metadata, and provenance contracts. It does **not** implement a beta-to-phi bridge, field potential, spatial operator, PDE, thermodynamics, gravity, quantisation, or cosmology.

## Scientific-status taxonomy

AgencityLab uses exactly four scientific statuses:

| Status | Meaning |
| --- | --- |
| `canonical` | Accepted definition of the Theory of Agencity and its reference implementation. |
| `experimental` | Direct computational extension intended to be evaluated without becoming a new canonical law. |
| `research` | Mathematical model proposed by the theory volumes and implemented for simulation or study without established empirical validation. |
| `speculative` | More strongly hypothetical theoretical extension, notably quantum agencity, agentons, and fundamental cosmological applications. |

Scientific status is not software maturity. It does not encode whether code is stable, well tested, production ready, or released. A speculative model may be implemented and tested rigorously while remaining scientifically speculative.

## Observable field versus autonomous dynamical field

Three names are intentionally distinct.

### `ObservableAgencityFieldResult`

This existing v1.1.0 result is constructed from an observable field `u(x,t)` by applying the canonical scalar temporal pipeline independently at each spatial location. Its principal outputs are `beta_obs(x,t)` and `b_obs(x,t)`.

Status: `experimental`.

The scalar quantities calculated locally remain canonical scalar quantities. What is experimental is the spatial orchestration of those local calculations. This object is not an autonomous field state and does not imply a field equation.

### `DynamicalAgencityFieldState`

This object represents one autonomous spatial snapshot `phi(x,t)` at a specified time. It may be real or complex and may optionally contain `phi_dot`.

Status: `research`.

It is not automatically an observable measured from `u(x,t)`. A future explicit bridge may connect an observable-field result to a dynamical field state, but that bridge must be named, documented, scientifically classified, and separately tested.

### `DynamicalAgencityFieldSolution`

This object represents a complete numerical trajectory of an autonomous field. The shape convention is:

```text
phi.shape == (n_time, *spatial_shape)
```

The optional `phi_dot` uses the same shape. `times` is one-dimensional and strictly increasing.

Status: `research`.

Do not introduce a generic `AgencityField` name for all three concepts. The existing v1.1 compatibility alias remains untouched until an integration or deprecation decision is made separately.

## Scientific progression

The accepted development progression is:

```text
u(x,t)
  -> beta_obs(x,t), b_obs(x,t)
  -> explicit research bridge
  -> phi(x,t)
  -> classical field dynamics
  -> coherent structures
  -> thermodynamics
  -> gravity
  -> quantum
  -> cosmology
```

The arrow `beta_obs -> phi` is **not canonical** and is **not implicit**. It is an explicit research bridge. This contract PR defines no formula for that bridge.

## Status by layer

| Layer | Status |
| --- | --- |
| scalar `u -> beta -> b` | `canonical` |
| observable spatial fields | `experimental` |
| numerical field discretization | `experimental` |
| autonomous classical `phi` | `research` |
| quartic potential / KG / TDGL | `research` |
| coherent structures | `research` |
| Agencity thermodynamics | `research` |
| classical gravity coupling | `research` |
| quantum field / agentons | `speculative` |
| cosmological application | `speculative` |

This classification does not prejudge the numerical outcome of simulations or future empirical validation.

## Units contract

Future field, gravity, quantum, and cosmology work starts in one of two explicit conventions:

- `dimensionless`
- `natural_units`

AgencityLab does not claim automatic conversion of these future field models to SI units. The observable scalar and observable-field layers retain their existing unit contracts.

Every `DynamicalAgencityFieldState`, `DynamicalAgencityFieldSolution`, and `FieldModelMetadata` must state its units convention explicitly.

## Parameter provenance

A numerical or physical parameter must not appear without an origin. The lightweight `ParameterProvenance` contract supports exactly these sources:

- `user_supplied`
- `named_physical_context`
- `dimensionless_benchmark`
- `source_document_reference`
- `derived_mathematically`
- `implementation_convention`

Example:

```python
ParameterProvenance(
    source="user_supplied",
    note="lambda provided by experiment configuration",
)
```

Parameters such as `lambda`, `mu`, `Gamma`, `a`, `T_c`, `T_eff`, `xi`, or later coefficients must come from the user, from an explicitly recorded physical context, or from a named and documented preset. There is no silent universal physical default.

The provenance model is deliberately small. It is not a dependency graph or symbolic derivation engine.

## Shared model metadata

`FieldModelMetadata` records compact scientific context without copying theory volumes into each result. It supports:

- `model_name`
- `scientific_status`
- `theory_source`
- `assumptions`
- `units_convention`
- `parameter_provenance`
- `software_version`
- `numerical_method`
- `boundary_condition`
- `grid_description`
- `notes`

Theory references should use stable textual identifiers such as a volume and chapter name. Do not invent equation numbers when the source does not provide a clear identifier.

Metadata intended for serialization must not contain Python callables. Large field arrays remain NumPy arrays in `to_dict()` rather than being automatically expanded into large Python lists. Full archival serialization can be added separately when a storage format is chosen.

## Shape and validation contracts

### State

For `DynamicalAgencityFieldState`:

- `phi.shape == spatial_shape`;
- `phi` must be finite;
- real input remains a real array and complex input remains complex;
- optional `phi_dot` must be finite and have the same shape;
- `time` must be finite;
- optional spatial axes must match `spatial_shape`;
- scientific status is `research`;
- units are `dimensionless` or `natural_units`.

The state contains no physical coefficient defaults.

### Solution

For `DynamicalAgencityFieldSolution`:

- `times` is one-dimensional, finite, and strictly increasing;
- `phi.shape == (len(times), *spatial_shape)`;
- optional `phi_dot` has the same shape;
- all field values are finite;
- optional spatial axes match `spatial_shape`;
- parameters and parameter provenance remain explicit;
- metadata and solver metadata must not contain Python callables;
- scientific status is `research`;
- units are `dimensionless` or `natural_units`.

These are data contracts only. They do not prescribe a solver, grid, boundary implementation, or dynamics equation.

## Two distinct Agencial entropy formulations

The theory volumes contain two distinct entropy formulations that must not be silently merged.

### Field agencial entropy

The field-theory thermodynamic formulation introduces an entropy proportional to

```text
(a / 2) * integral |phi|^2 dV
```

A future implementation should use a distinct name such as `field_agencial_entropy`.

### Contrast agencial entropy

A separate formulation in the observable theory uses the logarithmic contrast magnitude `|J|` relative to a `J_max` normalization. A future implementation should use a distinct name such as `contrast_agencial_entropy`.

This contract does not decide that one formula replaces the other. Their domains, assumptions, and scientific roles remain distinct until an explicit maintainer decision establishes a relationship.

## Governance for problematic formulas

A source inconsistency must never be silently repaired in code. Use this sequence:

```text
source formula
-> problem
-> analysis
-> proposed variant
-> maintainer decision
-> implementation
```

A PDF inconsistency must not be resolved only in a docstring, commit message, or hidden implementation choice. The decision must be traceable.

## Future dependencies

The following layers are future work and are not claimed to exist by this contract:

- explicit observable-to-dynamical bridge;
- classical field potentials and equations;
- coherent-structure solvers;
- Agencity thermodynamics;
- gravity coupling;
- quantum Agencity and agentons;
- cosmological applications.

The field-model contracts are intended to let these layers share names, scientific statuses, metadata, provenance, shapes, and units without duplicating or silently redefining scientific assumptions.
