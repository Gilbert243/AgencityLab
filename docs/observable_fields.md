# Observable Spatial Agencity Fields

**Status: experimental**

AgencityLab 1.1 adds an observable spatial extension without changing the canonical scalar theory. For a sampled field `u(x,t)` (or `u(x1,...,xd,t)`), every fixed spatial position is treated as an independent temporal observable:

```text
u(x,t)
  -> for each spatial location x: compute_agencity(u(x,.))
  -> beta_obs(x,t), b_obs(x,t)
```

The scalar pipeline executed locally is unchanged:

```text
u -> u* -> X* -> A* -> M,O -> D,S -> J,Theta,U -> beta -> b
```

The quantities returned at each location are therefore the canonical scalar quantities. What is experimental in v1.1 is the spatial orchestration, local-parameter interface, geometry metadata, and result model. This release does not claim new canonical field physics or empirical validation of a spatial field theory.

## Public API

```python
import numpy as np
from agencitylab.fields import compute_agencity_field

# u has shape (time, x, y)
t = np.arange(64, dtype=float)
x = np.linspace(0.0, 1.0, 4)
y = np.linspace(-1.0, 1.0, 3)
u = np.empty((t.size, x.size, y.size))
for i in range(x.size):
    for j in range(y.size):
        u[:, i, j] = np.sin(0.2 * t + 0.1 * i - 0.05 * j)

result = compute_agencity_field(
    u,
    t,
    spatial_axes=(x, y),
    A_ref=1.0,
    tau=4.0,
    w=3.0,
    P_c=2.0,
)

print(result.beta_obs.shape)
print(result.b_obs.shape)
```

`compute_agencity_field` and `ObservableAgencityFieldResult` are also exported from the top-level `agencitylab` namespace. The historical name `AgencityField` is retained only as a compatibility alias for `ObservableAgencityFieldResult`; it is not a dynamical field type.

## Temporal CRM only

**CRM is evaluated along the temporal axis independently at each spatial location. No spatial CRM is introduced by v1.1.**

No neighbouring spatial samples are correlated or smoothed by this API. The memory and organisation fields are simply the local scalar results:

```text
M(x,t) = CRM_t[u*(x,.)](t)
O(x,t) = CRM_t[u*(x,.), X*(x,.)](t)
```

The finite-record CRM warm-up semantics are inherited exactly from `compute_agencity()`. v1.1 does not alter `M`, `O`, `S`, `beta`, or `b` during warm-up and does not introduce a separate physical validity mask.

## Shapes and spatial geometry

The observable may have arbitrary spatial dimensionality. With `time_axis=0`, the conventional shape is:

```text
(n_time, n1, ..., nd)
```

A configurable `time_axis` is supported, including negative NumPy-style indices. Internally the temporal axis is moved to the front, spatial dimensions are flattened, scalar trajectories are computed independently, and all result arrays are restored to the original geometry.

`spatial_axes` is optional. If omitted, each spatial coordinate is recorded using sampled integer indices. If supplied, it must contain exactly one finite, one-dimensional, strictly monotone coordinate array per spatial dimension, with the matching dimension length. These coordinates are geometry metadata only; the observable pipeline takes no spatial derivative in v1.1.

## Local physical parameters

The API resolves shapes explicitly rather than relying on accidental NumPy broadcasting.

| Parameter | Scalar | Spatial field | Spacetime field |
| --- | --- | --- | --- |
| `A_ref` | yes | exact `spatial_shape` | no |
| `tau` | yes | exact `spatial_shape` | no |
| `w` | yes | exact `spatial_shape` | no |
| `P_c` | yes | exact `spatial_shape` | exact `u.shape` |

All values must be finite. `A_ref`, `tau`, and `w` must be strictly positive. `P_c` must be non-negative; exact local zero is valid and gives `b_obs=0` without changing `beta_obs`.

`A_ref` is a physical/contextual reference amplitude and is never estimated from standard deviation, variance, MAD, range, z-score, or another statistic of `u`.

### `tau` and `w` remain distinct

An explicitly supplied `w(x)` is preserved independently from `tau(x)`. If `w` is omitted, v1.1 uses the software fallback:

```text
w(x) = tau(x)
```

and records:

```text
w was unspecified; implementation convention w = tau was used
```

This is an implementation convention, not a physical identity.

## Result model

`ObservableAgencityFieldResult` stores:

- `t`, `spatial_axes`, `time_axis`, and `spatial_shape`;
- `u`, `u_star`, `X_star`, `A_star`;
- `M`, `O`, `D`, `S`, `J`, `U`;
- `beta` / `beta_obs` and `b` / `b_obs`;
- resolved `A_ref`, `tau`, `w`, and `P_c` arrays;
- reproducibility metadata, the producing AgencityLab version, and backend;
- `status="experimental"` and `model="observable_agencity_field"`.

NumPy is the v1.1 reference implementation. No field-specific Numba, JAX, GPU, SciPy, pandas, xarray, or Matplotlib dependency is required.

## Observable field versus dynamical field

The v1.1 object is an **Observable Agencity Field** computed from measurements `u(x,t)`. It is not the future autonomous **Dynamical Agencity Field** `phi(x,t)`.

The following are deliberately **not implemented in v1.1**:

- autonomous `phi(x,t)` dynamics;
- potentials or variational equations;
- PDE evolution or Laplacian coupling;
- Klein-Gordon or dissipative Klein-Gordon equations;
- TDGL or Ginzburg-Landau dynamics;
- boundary-condition solvers;
- domain walls or vortices;
- coherent-structure or phase-transition solvers;
- thermodynamics or Landauer extensions;
- gravity, Einstein equations, quantum fields, agentons, or cosmology.

Historical modules under `agencitylab/fields/` that previously returned placeholder PDE, solver, boundary, energy, action, or domain-wall values are explicitly non-operational in 1.1. They are reserved for a scientifically defined v1.2 research implementation rather than being treated as accepted physics.

## Scientific boundary

v1.1 implements **same canonical theory, evaluated locally over space**. Pointwise equivalence to independent `compute_agencity()` calls is the defining implementation property and is covered by tests for homogeneous and heterogeneous parameters, multidimensional geometries, time-axis permutations, exact rest, local `P_c=0`, invariances, and spatio-temporal power profiles.
