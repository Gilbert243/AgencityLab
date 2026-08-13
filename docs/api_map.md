# Scientific API map

AgencityLab 1.1.7 uses a small package top level and explicit scientific namespaces. The organization is architectural only: **no scientific equation is changed by this API map**.

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

`compute_agencity()` is the sole reference scalar orchestration. Physical/contextual inputs are explicit and are not injected from global configuration.

## I want to analyse a result

**Status: diagnostic; does not redefine the canonical state**

```python
import agencitylab.analysis as analysis

summary = analysis.analyze_agencity(result)
```

Named coherence, events, transitions, regimes, geometry, inverse-signature and related helpers live under `agencitylab.analysis` or the stable workflow namespace `agencitylab.api`.

## I want an observable spatial field

**Status: experimental**

```python
import agencitylab.fields as fields

field = fields.compute_agencity_field(...)
```

This applies the scalar temporal pipeline locally in space. It does not automatically promote observable `beta_obs` to an autonomous field.

## I want to study the autonomous field

**Status: research**

```python
import agencitylab.fields as fields

potential = fields.QuarticAgencityPotential(lambda_=..., mu=...)
solution = fields.simulate_klein_gordon(...)
```

The bridge `phi = sqrt(P_c * tau) * beta` remains explicit. Observable `beta_obs` and autonomous `phi` are distinct scientific quantities.

## Thermodynamics

**Status: research**

```python
import agencitylab.thermodynamics as thermodynamics
```

Field and contrast agencial entropies remain distinct APIs. No global temperature is supplied by runtime configuration.

## Gravity

**Status: research**

```python
import agencitylab.gravity as gravity
```

Gravity explicitly uses the Chapter-19 `(-,+,+,+)` convention. It is not silently unified with the Chapter-16 flat-field `(+,-,-,-)` convention.

## Quantum

**Status: speculative**

```python
import agencitylab.quantum as quantum
```

These are computational primitives for the proposed autonomous-field quantization, not experimental evidence for quantum Agencity.

## Cosmology

**Status: speculative**

```python
from agencitylab.applications import cosmology
```

The cosmology package is a homogeneous reference application, not an observational inference framework.

## Stable workflows and utilities

Batch, streaming, reports, exports, visualization, pipelines, discrete/multivariate calculations and scientific workflows remain grouped under:

```python
import agencitylab.api as api
```

They orchestrate existing computation and analysis; they are not independent canonical equations.

## Top-level compatibility

Earlier 1.1.x releases exposed many specialized functions directly at package top level. In 1.1.7 those names remain lazy compatibility aliases, but explicit use emits `DeprecationWarning` with the recommended namespace. Plain `import agencitylab` emits no warning.

The recommended top-level surface is intentionally small: version, scientific status, principal result models and exceptions, `compute_agencity`, `compute_agencity_field`, `analyze_agencity`, `scientific_workflow`, plus the scientific namespaces themselves.

## Legacy dynamics

`agencitylab.dynamics.system` is not canonical theory. The theory states `beta = J * U` and `b = P_c * beta`, while the historical module used tanh-based beta factors and a discrete beta variation. The misleading equations are retired; generic numerical/dynamical-system helpers with independent software value remain.

## Runtime configuration boundary

`agencitylab.config` contains software/runtime options only. Historical keys such as `crm_window`, `temperature`, `agencity_scale`, `epsilon`, and direct physical-parameter keys are accepted only as deprecated compatibility metadata and cannot alter scientific equations.

Physical/contextual quantities belong to the scientific call that consumes them, including `A_ref`, `tau`, `w`, `P_c`, `Gamma`, `lambda`, `mu`, thermal parameters, `xi`, and `G`.

## Software status versus scientific status

The package retains the PyPI `Production/Stable` classifier because the documented software contract is stable. That classifier is not a claim that experimental, research, or speculative layers are empirically validated.
