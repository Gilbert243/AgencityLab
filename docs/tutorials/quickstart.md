# Quickstart: from signal to scientific result

This tutorial uses explicit physical/contextual inputs and produces a result, diagnostics, figures, and machine-readable exports. Install the visualization extra first:

```bash
pip install "agencitylab[viz]"
```

## 1. Create a reproducible signal

```python
import numpy as np

xi = np.linspace(0.0, 40.0, 1601)
u = np.sin(xi) + 0.15 * np.sin(2.0 * xi)
```

No noise is added here, so the example is deterministic.

## 2. Compute the Theory of Agencity quantities

```python
from agencitylab import compute_agencity

result = compute_agencity(
    u=u,
    xi=xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
    unit="rad",
    coordinate_unit="s",
    power_unit="W",
)
```

`tau` and `w` are distinct in Volume 2 of the theory. If `w` is omitted, AgencityLab uses the common convention `w=tau`; an explicit positive `w` is preserved.

The result exposes the whole computational chain:

```python
print(result.X_star.shape)
print(result.M.shape, result.O.shape)
print(result.D.shape, result.S.shape)
print(result.J.shape, result.theta.shape)
print(result.beta.shape, result.b.shape)
print(result.summary())
```

## 3. Build diagnostics

```python
from agencitylab import analyze_agencity

analysis = analyze_agencity(result)
print(analysis["real_agencity"]["status"])
print(analysis["regime"])
```

For a non-null signal these interpretations normally remain `undetermined` until the experiment supplies contextual thresholds or regime criteria. AgencityLab does not invent universal thresholds for “low” angular variance or “significant” `|b|`.

## 4. Make scientific figures

```python
from agencitylab import visualize_agencity

visualize_agencity(result, kind="overview")
visualize_agencity(result, kind="geometry", analysis=analysis)
visualize_agencity(result, kind="diagnostics", analysis=analysis)
```

The geometry view uses the intrinsic complex `beta` trajectory, not `b`, so an externally varying `P_c(t)` does not redefine intrinsic state geometry.

## 5. Export CSV and JSON

```python
from agencitylab import export_result_csv, export_study_json, textual_analysis

report = textual_analysis(result)
export_result_csv(result, "agencity_result.csv")
export_study_json(
    result,
    analysis,
    "agencity_study.json",
    text_report=report,
)
```

The CSV has one row per sample and explicit real/imaginary columns for `beta` and `b`. The JSON bundle keeps the stable result serialization plus the structured diagnostic report.

## One-call workflow

The same workflow can be orchestrated with:

```python
from agencitylab import scientific_workflow

study = scientific_workflow(
    u,
    xi,
    A_ref=1.0,
    tau=2.0,
    w=1.5,
    P_c=5.0,
    coordinate_unit="s",
    power_unit="W",
    export_dir="agencity_output",
)

print(study.report)
print(study.exports)
```

This is an orchestration convenience: it does not replace or alter any theory equation.
