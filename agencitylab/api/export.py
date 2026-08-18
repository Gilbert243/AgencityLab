"""Research-facing export API for AgencityLab.

CSV exports of :class:`AgencityResult` are sample-wise scientific tables.
JSON exports preserve the stable result serialization contract and may bundle
structured diagnostics for reproducible studies.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from agencitylab.io.csv import dump_csv
from agencitylab.io.json import dump_json

SCIENTIFIC_UX_SCHEMA_VERSION = "1.0"


def _ensure_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _flatten_for_table(data: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested report values for generic table exporters."""
    if hasattr(data, "to_dict") and callable(data.to_dict):
        data = data.to_dict()
    elif is_dataclass(data):
        data = asdict(cast(Any, data))

    items: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{parent_key}{sep}{key}" if parent_key else str(key)
            items.update(_flatten_for_table(value, name, sep=sep))
        return items
    if isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            name = f"{parent_key}{sep}{index}" if parent_key else str(index)
            items.update(_flatten_for_table(value, name, sep=sep))
        return items
    if isinstance(data, np.ndarray):
        return {parent_key: data.tolist() if parent_key else data.tolist()}
    if isinstance(data, (np.integer, np.floating)):
        return {parent_key: data.item()}
    if isinstance(data, complex):
        return {parent_key: {"real": float(data.real), "imag": float(data.imag)}}
    return {parent_key: data}


def export_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Export a result, report, or JSON-compatible study payload."""
    return dump_json(data, _ensure_path(path), indent=indent, sort_keys=sort_keys)


def export_result_csv(result: Any, path: str | Path) -> Path:
    """Export one row per sample with canonical scalar quantities and complex parts."""
    if not hasattr(result, "to_dataframe"):
        raise TypeError("export_result_csv expects an AgencityResult-like object")
    output = _ensure_path(path)
    frame = result.to_dataframe()
    frame.to_csv(output, index=False)
    return output


def export_csv(data: Any, path: str | Path) -> Path:
    """Export a result as a sample table, or a generic report as flattened CSV."""
    if hasattr(data, "to_dataframe") and callable(data.to_dataframe):
        return export_result_csv(data, path)
    return dump_csv(data, _ensure_path(path))


def export_study_json(
    result: Any,
    analysis: dict[str, Any],
    path: str | Path,
    *,
    text_report: str | None = None,
) -> Path:
    """Export a reproducible canonical-result + diagnostics bundle."""
    payload = {
        "scientific_ux_schema_version": SCIENTIFIC_UX_SCHEMA_VERSION,
        "result": result.to_dict() if hasattr(result, "to_dict") else result,
        "analysis": analysis,
        "text_report": text_report,
    }
    return export_json(payload, path)


def export_excel(data: Any, path: str | Path) -> Path:
    """Export a result or report dictionary to Excel (.xlsx)."""
    output = _ensure_path(path)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for Excel export; install agencitylab[export]") from exc
    if hasattr(data, "to_dataframe") and callable(data.to_dataframe):
        data.to_dataframe().to_excel(output, index=False)
    else:
        pd.DataFrame([_flatten_for_table(data)]).to_excel(output, index=False)
    return output


def export_pdf(text_report: str, path: str | Path) -> Path:
    """Export a text report to PDF using the optional reportlab dependency."""
    output = _ensure_path(path)
    try:
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Preformatted, SimpleDocTemplate
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("reportlab is required for PDF export; install agencitylab[export]") from exc
    doc = SimpleDocTemplate(str(output))
    styles = getSampleStyleSheet()
    doc.build([Preformatted(str(text_report), styles["Code"])])
    return output


def export_report(
    report: Any,
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    """Export a report using a format inferred from the suffix when omitted."""
    output = Path(path)
    fmt = (format or output.suffix.lstrip(".")).lower()
    if fmt == "json":
        return export_json(report, output)
    if fmt == "csv":
        return export_csv(report, output)
    if fmt in {"xlsx", "xls", "excel"}:
        return export_excel(report, output)
    if fmt == "pdf":
        if not isinstance(report, str):
            from agencitylab.api.report import build_text_report

            report = build_text_report(report)
        return export_pdf(report, output)
    raise ValueError(f"Unsupported export format: {fmt}")
