# Scientific visualization gallery

Install the visualization extra:

```bash
pip install "agencitylab[viz]"
```

All v0.7 figures consume an `AgencityResult` or an already-computed analysis. Plotting does not modify the Theory of Agencity quantities.

## Scientific overview

```python
from agencitylab import visualize_agencity

fig = visualize_agencity(result, kind="overview", show=False)
fig.savefig("overview.png", dpi=200, bbox_inches="tight")
```

The overview shows the observable, reduced kinematics, CRM pair `(M,O)`, intensities/contrast, intrinsic state, observable flux, and structural orientation.

## Intrinsic beta geometry

```python
fig = visualize_agencity(
    result,
    kind="geometry",
    analysis=analysis,
    show=False,
)
```

The left panel is the complex trajectory `(Re(beta), Im(beta))`. The right panel displays signed beta curvature. Geometry is intrinsic to `beta`; it is not computed from `b=P_c beta`.

## Real-agencity diagnostics

```python
fig = visualize_agencity(
    result,
    kind="diagnostics",
    analysis=analysis,
    show=False,
)
```

This view includes `S`, local `Sigma_Theta`, `|b|`, and dynamic/structural contrast. Threshold lines appear only if the supplied analysis already contains contextual thresholds.

## Explicit complex time series

```python
visualize_agencity(result, kind="timeseries")
```

`beta` and `b` are complex. The compatibility time-series view therefore plots their real part, imaginary part, and magnitude separately instead of silently casting complex values to real values.

## Component view

```python
visualize_agencity(result, kind="components")
```

This plots `X*`, `A*`, `M=CRM[u*]`, and `O=CRM[u*,X*]` with explicit labels.

## Frequency spectrum

```python
visualize_agencity(result, kind="frequency_spectrum", component="magnitude")
```

This is a descriptive Fourier diagnostic of a chosen component of the observable flux. It is distinct from the theoretical multiscale `b(t,tau)` spectrum.

## Multiscale b(t, tau)

```python
from agencitylab import compute_agencity_spectrum, visualize_multiscale_spectrum

spectrum = compute_agencity_spectrum(
    result.u,
    result.xi,
    taus=[1.0, 2.0, 4.0],
    A_ref=result.A_ref,
    P_c=5.0,
    return_full=True,
)
visualize_multiscale_spectrum(spectrum)
```

The heatmap represents `|b(t,tau)|`. It should not be confused with the Fourier-frequency spectrum.

## Canonical component heatmap

```python
visualize_agencity(result, kind="heatmap")
```

The compatibility heatmap displays the dimensionless trajectories `M`, `O`, `D`, `S`, and `J` on a common time axis.

## Publication use

Every plotting function returns the Matplotlib `Figure`, allowing the researcher to choose file format, DPI, journal dimensions, or downstream annotation without changing the numerical result:

```python
fig = visualize_agencity(result, kind="geometry", analysis=analysis, show=False)
fig.savefig("figure_2.svg", bbox_inches="tight")
```
