"""End-to-end researcher workflow for AgencityLab v0.7."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .analyze import analyze_agencity, textual_analysis
from .compute import compute_agencity
from .export import export_result_csv, export_study_json
from .visualize import visualize_agencity


@dataclass(slots=True)
class ScientificStudy:
    """Artifacts produced by a reproducible signal-to-diagnostic workflow."""

    result: Any
    analysis: dict
    report: str
    figures: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Path] = field(default_factory=dict)


def scientific_workflow(
    u,
    xi=None,
    *,
    A_ref,
    tau,
    P_c,
    w=None,
    analysis_kwargs: Mapping[str, Any] | None = None,
    figure_kinds: Iterable[str] = ("overview", "diagnostics", "geometry"),
    show: bool = False,
    export_dir: str | Path | None = None,
    **compute_kwargs,
) -> ScientificStudy:
    """Run ``signal -> result -> diagnostics -> report -> figures``.

    Physical/contextual quantities are explicit. Diagnostic thresholds, when
    needed, must be supplied in ``analysis_kwargs``; this workflow never invents
    universal real-agencity or regime thresholds. If ``export_dir`` is supplied,
    the canonical sample table, reproducible study JSON, text report and created
    figures are written there.
    """
    result = compute_agencity(
        u=u,
        xi=xi,
        A_ref=A_ref,
        tau=tau,
        w=w,
        P_c=P_c,
        **compute_kwargs,
    )
    diagnostic_options = dict(analysis_kwargs or {})
    analysis = analyze_agencity(result, **diagnostic_options)
    report = textual_analysis(result, **diagnostic_options)
    result.attach_analysis(analysis)
    result.attach_report(report)

    figures: dict[str, Any] = {}
    for kind in tuple(figure_kinds):
        kwargs = {"analysis": analysis} if kind in {"diagnostics", "geometry"} else {}
        figures[str(kind)] = visualize_agencity(result, kind=str(kind), show=show, **kwargs)

    exports: dict[str, Path] = {}
    if export_dir is not None:
        directory = Path(export_dir)
        directory.mkdir(parents=True, exist_ok=True)
        exports["csv"] = export_result_csv(result, directory / "agencity_result.csv")
        exports["json"] = export_study_json(
            result,
            analysis,
            directory / "agencity_study.json",
            text_report=report,
        )
        report_path = directory / "agencity_report.txt"
        report_path.write_text(report, encoding="utf-8")
        exports["report"] = report_path
        for kind, figure in figures.items():
            path = directory / f"agencity_{kind}.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            exports[f"figure_{kind}"] = path

    return ScientificStudy(
        result=result,
        analysis=analysis,
        report=report,
        figures=figures,
        exports=exports,
    )
