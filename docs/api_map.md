# Scientific API map

AgencityLab 1.0 uses a small package root and explicit scientific namespaces.
This organization is architectural only: **no scientific equation is changed by
this API map**.

## I want to calculate Agencity for a signal

**Status: canonical**

```python
import agencitylab as al

result = al.compute_agencity(
    u,
    A_ref=...,
    tau=...,
    w=...,
    P_c=...,
)
```

`compute_agencity()` is the sole reference scalar orchestration. Physical and
contextual inputs are explicit and are not injected from global software
configuration.

## I want to analyse a result

**Status: diagnostic; does not redefine the canonical state**

```python
import agencitylab.analysis as analysis

summary = analysis.analyze_agencity(result)
```

Named coherence, events, transitions, regimes, geometry, inverse-signature and
related helpers live under `agencitylab.analysis` or the stable workflow
namespace `agencitylab.api`.

Diagnostic products are separate from the canonical `AgencityResult`. High-level
workflow objects may carry analysis, reports and figures without mutating the
canonical result model.

## I want an observable spatial field

**Status: experimental**

```python
import agencitylab.fields as fields

field = fields.compute_agencity_field(...)
```

This applies the scalar temporal pipeline locally in space. It does not
automatically promote observable `beta_obs` to an autonomous field.

## I want to study the autonomous field

**Status: research**

```python
import agencitylab.fields as fields

potential = fields.QuarticAgencityPotential(lambda_=..., mu=...)
solution = fields.simulate_klein_gordon(...)
```

The bridge `phi = sqrt(P_c * tau) * beta` remains explicit. Observable
`beta_obs` and autonomous `phi` are distinct scientific quantities.

## Thermodynamics

**Status: research**

```python
import agencitylab.thermodynamics as thermodynamics
```

Field and contrast agencial entropies remain distinct APIs. No global
temperature is supplied by runtime configuration.

## Gravity

**Status: research**

```python
import agencitylab.gravity as gravity
```

Gravity explicitly uses the Chapter-19 `(-,+,+,+)` convention. It is not
silently unified with the Chapter-16 flat-field `(+,-,-,-)` convention.

## Quantum

**Status: speculative**

```python
import agencitylab.quantum as quantum
```

These are computational primitives for the proposed autonomous-field
quantization, not experimental evidence for quantum Agencity.

## Cosmology

**Status: speculative**

```python
from agencitylab.applications import cosmology
```

The cosmology package is a homogeneous reference application, not an
observational inference framework.

## Stable workflows and utilities

Batch, streaming, reports, exports, visualization, pipelines,
discrete/multivariate calculations and scientific workflows live under:

```python
import agencitylab.api as api
```

They orchestrate existing computation and analysis; they are not independent
canonical equations.

## Package root

The recommended package-root surface is intentionally small: version,
scientific status, principal result models and exceptions,
`compute_agencity`, `compute_agencity_field`, `analyze_agencity`,
`scientific_workflow`, plus the primary scientific namespaces.

AgencityLab 1.0 is the first stable public API contract. Development snapshots
that preceded 1.0 do not create package-root compatibility aliases.

## Generic dynamics

`agencitylab.dynamics` contains scientifically neutral dynamical-systems
utilities such as attractor, bifurcation, delay, stability and integration
helpers. It does not define an alternative Agencity state or an alternative
`beta`/`b` construction.

The incorrect pre-1.0 tanh-based Agencity dynamical model is not part of the 1.0
package.

## Runtime configuration boundary

`agencitylab.config` contains software/runtime options only. Unknown options are
errors rather than hidden metadata.

Physical/contextual quantities belong to the scientific call that consumes
them, including `A_ref`, `tau`, `w`, `P_c`, `Gamma`, `lambda`, `mu`, thermal
parameters, `xi`, and `G`.

## Result and serialization boundary

`AgencityResult` is the canonical scalar result model. Result serialization and
optional pandas/xarray adapters are implemented under `agencitylab.io` and are
exposed through thin convenience methods on the result object.

The stable result schema is `1.0`. Development-only payload schemas are not
implicitly migrated by the stable 1.0 deserializer.

## Software status versus scientific status

The PyPI `Production/Stable` classifier describes the documented software
contract. It is not a claim that experimental, research, or speculative layers
are empirically validated.
