# v0.7 — Scientific UX

Version 0.7 makes AgencityLab usable as a research workflow without requiring the user to read internal implementation modules. The design goal is a transparent chain:

```text
signal -> theoretical result -> diagnostics -> report -> figure -> export
```

The UX layer is orchestration and presentation. It does not redefine `M`, `O`, `D`, `S`, `J`, `Theta`, `beta`, or `b`.

## Installation

Core numerical use:

```bash
pip install agencitylab
```

Scientific figures:

```bash
pip install "agencitylab[viz]"
```

Excel/PDF compatibility exports use the optional `export` extra. CSV and JSON do not require it.

## One theory, two volumes

AgencityLab treats the two theory documents as two volumes of the same Theory of Agencity. When Volume 2 specifies or generalises a construction, that definition governs the implementation.

A v0.7 audit exposed one important software mismatch inherited from v0.6: the public scalar API had enforced `w=tau`, whereas Volume 2 keeps the CRM width `w>0` distinct and says that `w=tau` is often a convenient convention. v0.7 corrects the API:

```python
result = compute_agencity(..., tau=2.0)          # default w=tau
result = compute_agencity(..., tau=2.0, w=1.0)  # explicit w
```

This correction changes parameter handling, not the formulas for CRM, intensities, contrast, orientation, state, or flux.

## Computation

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

The result object stores the full sample-wise pipeline and scientific metadata. `A_ref`, `tau`, `w`, and `P_c` remain explicit physical/contextual inputs unless a documented theory-defined procedure is invoked separately.

## Diagnostics

```python
from agencitylab import analyze_agencity

analysis = analyze_agencity(result)
```

Important sections include:

- `metrics`: magnitude, variance, energy and related summaries;
- `coherence.structural_orientation`: structural phase statistics and `Sigma_Theta`;
- `geometry`: signed curvature of intrinsic `beta` and winding diagnostics;
- `events`: peaks of `D`;
- `transitions`: zeros, `D=S` crossings, and optional `Theta` jumps;
- `structural_plateaus`: optional `S` plateau diagnostics;
- `regime_signature`: threshold-free descriptive features;
- `real_agencity`: the separate real-agencity diagnostic.

No universal values are inserted for “low `Sigma_Theta`” or “significant `|b|`.” Without contextual thresholds, a non-null real-agencity result remains `undetermined`.

The derived indicator `Sigma_Theta` continues to use its theoretical interval `[t-tau,t]`. CRM finite-record warm-up, by contrast, depends on the CRM window and begins after two complete windows, `t >= t0 + 2w`.

## Reports

```python
from agencitylab import textual_analysis

report = textual_analysis(result)
print(report)
```

The human-readable report includes sample count, `tau`, `w`, `A_ref`, mean `|b|`, mean `J`, mean `Sigma_Theta`, mean absolute beta curvature, winding, regime status, and real-agencity status.

For structured downstream processing, keep the dictionary from `analyze_agencity()` rather than parsing the text report.

## Scientific figures

```python
from agencitylab import visualize_agencity

fig = visualize_agencity(result, kind="overview", show=False)
```

Recommended views:

- `overview`: computational stages from `u` to `b`;
- `geometry`: intrinsic beta-plane trajectory and curvature;
- `diagnostics`: `S`, `Sigma_Theta`, `|b|`, and dynamic/structural contrast.

Compatibility views remain available for time series, components, frequency spectrum, and heatmaps. Complex quantities are displayed explicitly as real/imaginary/magnitude components where appropriate.

The Fourier-frequency spectrum and the theoretical multiscale `b(t,tau)` spectrum are different objects. Use:

```python
from agencitylab import visualize_multiscale_spectrum
visualize_multiscale_spectrum(spectrum)
```

for a result produced by `compute_agencity_spectrum()`.

## CSV export

```python
from agencitylab import export_result_csv

export_result_csv(result, "result.csv")
```

The table contains one row per sample. Core columns include:

```text
xi, u, u_star, X_star, A_star,
M, O, D, S, J, theta, P_c,
beta_real, beta_imag, beta_abs,
b_real, b_imag, b_abs
```

Splitting complex values into explicit columns avoids ambiguous complex-number formatting in generic spreadsheet/statistical tools.

## Reproducible JSON bundle

```python
from agencitylab import export_study_json, textual_analysis

report = textual_analysis(result)
export_study_json(result, analysis, "study.json", text_report=report)
```

The bundle contains:

- `scientific_ux_schema_version`;
- the stable `AgencityResult` serialization;
- the structured diagnostic dictionary;
- optionally the text report.

The result schema remains `0.3` because v0.7 does not alter the numerical result model.

## One-call researcher workflow

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
    export_dir="study_output",
    show=False,
)
```

The returned `ScientificStudy` contains:

```text
study.result
study.analysis
study.report
study.figures
study.exports
```

This makes each stage inspectable. A user may rerun diagnostics with different contextual criteria without recomputing the theoretical result.

## Reproducibility policy

A research example should record at minimum:

- the input signal or its deterministic generation procedure;
- coordinate definition and sampling interval;
- `A_ref`, `tau`, `w`, `P_c`, and their units/context;
- AgencityLab version;
- any diagnostic thresholds or `RegimeCriteria` and why they were selected;
- any preprocessing performed before AgencityLab;
- the result/analysis JSON and, when useful, sample-wise CSV.

Random examples should use an explicit seed and physically filtered stochastic processes when derivatives are interpreted scientifically. Ideal white noise is not silently treated as a differentiable physical signal.

## Scientific interpretation boundary

A figure is not an additional equation. A CSV column is not a new observable. A report label is not a new physical law. The v0.7 UX layer exposes the theory and diagnostic outputs while preserving the distinctions already established in the computational and analysis layers.

`beta != 0` still does not establish coherent real agencity. High `D` still does not imply real agencity. Noise and chaos may produce local non-zero `beta` and intermittent local diagnostic satisfaction.
