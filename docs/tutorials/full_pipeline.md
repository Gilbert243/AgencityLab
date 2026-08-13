# Full scientific workflow

This tutorial follows the intended v0.7 path:

```text
signal -> AgencityResult -> diagnostics -> report -> figure -> export
```

The computation layer and the diagnostic layer remain separate. Diagnostic thresholds do not alter `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b`.

## Reproducible input

```python
import numpy as np

xi = np.linspace(0.0, 60.0, 2401)
u = np.sin(xi) + 0.2 * np.sin(0.5 * xi)
```

For measured data, replace these arrays with calibrated samples and keep the physical metadata needed to justify `A_ref`, `tau`, `w`, and `P_c`.

## Step 1 — computation

```python
from agencitylab import compute_agencity

result = compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.5,
    w=2.0,
    P_c=10.0,
    unit="rad",
    coordinate_unit="s",
    power_unit="W",
)
```

`w` is the CRM width and `tau` is the characteristic structural time. Volume 2 keeps them distinct. Omitting `w` selects the common convention `w=tau`; supplying it makes the theory parameter explicit.

## Step 2 — diagnostics without interpretation thresholds

```python
from agencitylab import analyze_agencity

analysis = analyze_agencity(result)

print(analysis["metrics"]["mean_magnitude"])
print(analysis["coherence"]["structural_orientation"]["sigma_theta_mean"])
print(analysis["geometry"]["curvature_mean_abs"])
print(analysis["geometry"]["winding"])
print(analysis["real_agencity"]["status"])
```

The last status is intentionally `undetermined` when contextual real-agencity thresholds have not been supplied.

## Step 3 — contextual real-agencity diagnostic

Only supply thresholds when the experiment or domain justifies them. The following numbers are **illustrative placeholders**, not Theory of Agencity constants:

```python
analysis_with_context = analyze_agencity(
    result,
    real_agencity_thresholds={
        "theta_variance_threshold": 0.05,  # illustrative only
        "b_threshold": 0.5,               # illustrative only, in the result b unit
        "min_fraction": 0.8,              # illustrative persistence rule
    },
)
```

A publication or benchmark should record the origin of such thresholds.

## Step 4 — human-readable report

```python
from agencitylab import textual_analysis

report = textual_analysis(result)
print(report)
```

The report records `tau`, CRM width `w`, `A_ref`, mean `|b|`, `J`, `Sigma_Theta`, beta curvature, winding, regime status, and real-agencity status.

## Step 5 — scientific figures

```python
from agencitylab import visualize_agencity

fig_overview = visualize_agencity(result, kind="overview", show=False)
fig_geometry = visualize_agencity(
    result,
    kind="geometry",
    analysis=analysis,
    show=False,
)
fig_diagnostics = visualize_agencity(
    result,
    kind="diagnostics",
    analysis=analysis,
    show=False,
)

fig_overview.savefig("overview.png", dpi=200, bbox_inches="tight")
fig_geometry.savefig("geometry.png", dpi=200, bbox_inches="tight")
fig_diagnostics.savefig("diagnostics.png", dpi=200, bbox_inches="tight")
```

The overview follows the computational stages. The geometry view is based on intrinsic `beta`. The diagnostics view consumes the already-computed analysis and does not create thresholds on its own.

## Step 6 — machine-readable exports

```python
from agencitylab import export_result_csv, export_study_json

export_result_csv(result, "result.csv")
export_study_json(result, analysis, "study.json", text_report=report)
```

Use the CSV for sample-wise numerical work. Use the JSON bundle when you need the result metadata and structured diagnostics together.

## Equivalent high-level workflow

```python
from agencitylab import scientific_workflow

study = scientific_workflow(
    u,
    xi,
    A_ref=1.0,
    tau=2.5,
    w=2.0,
    P_c=10.0,
    unit="rad",
    coordinate_unit="s",
    power_unit="W",
    export_dir="study_output",
    show=False,
)
```

`study.result`, `study.analysis`, `study.report`, `study.figures`, and `study.exports` expose each stage separately, so the researcher can inspect or replace any diagnostic step without modifying the computation.

## Multiscale continuation

```python
from agencitylab import compute_agencity_spectrum, visualize_multiscale_spectrum

spectrum = compute_agencity_spectrum(
    u,
    xi,
    taus=[1.0, 2.0, 4.0, 8.0],
    A_ref=1.0,
    P_c=10.0,
    windows=2.0,
    return_full=True,
)

visualize_multiscale_spectrum(spectrum)
```

The resulting `b(t, tau)` view is a scale study. A peak in it is not automatically the physical characteristic time.
