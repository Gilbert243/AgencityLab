"""Research-facing export API for AgencityLab.

CSV exports of :class:`AgencityResult` are sample-wise scientific tables.
JSON exports preserve the result serialization contract and can bundle the
structured diagnostic report for reproducible studies.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from agencitylab.io.csv import dump_csv
from agencitylab.io.json import dump_json

SCIENTIFIC_UX_SCHEMA_VERSION = "0.7"


def _ensure_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _flatten_for_table(data: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested values for the historical report-table exporters."""
    if hasattr(data, "to_dict") and callable(data.to_dict):
        data = data.to_dict()
    elif is_dataclass(data):
        data = asdict(data)

    items: Dict[str, Any] = {}
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
    path: Union[str, Path],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Export a result, report, or JSON-compatible study payload."""
    return dump_json(
        data,
        _ensure_path(path),
        indent=indent,
        sort_keys=sort_keys,
    )


def export_result_csv(result, path: Union[str, Path]) -> Path:
    """Export one row per sample with canonical scalar quantities and complex parts."""
    if not hasattr(result, "to_dataframe"):
        raise TypeError("export_result_csv expects an AgencityResult-like object")
    path = _ensure_path(path)
    frame = result.to_dataframe()
    frame.to_csv(path, index=False)
    return path


def export_csv(data: Any, path: Union[str, Path]) -> Path:
    """Export a result as a sample table, or a generic report as flattened CSV."""
    if hasattr(data, "to_dataframe") and callable(data.to_dataframe):
        return export_result_csv(data, path)
    return dump_csv(data, _ensure_path(path))


def export_study_json(
    result,
    analysis: dict,
    path: Union[str, Path],
    *,
    text_report: str | None = None,
) -> Path:
    """Export a reproducible result + diagnostics bundle without figure objects."""
    payload = {
        "scientific_ux_schema_version": SCIENTIFIC_UX_SCHEMA_VERSION,
        "result": result.to_dict() if hasattr(result, "to_dict") else result,
        "analysis": analysis,
        "text_report": text_report,
    }
    return export_json(payload, path)


def export_excel(data: Any, path: Union[str, Path]) -> Path:
    """Export a report dictionary to Excel (.xlsx)."""
    path = _ensure_path(path)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for Excel export") from exc
    if hasattr(data, "to_dataframe") and callable(data.to_dataframe):
        data.to_dataframe().to_excel(path, index=False)
    else:
        pd.DataFrame([_flatten_for_table(data)]).to_excel(path, index=False)
    return path


def export_pdf(text_report: str, path: Union[str, Path]) -> Path:
    """Export a text report to PDF using the optional reportlab dependency."""
    path = _ensure_path(path)
    try:
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Preformatted, SimpleDocTemplate
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("reportlab is required for PDF export") from exc
    doc = SimpleDocTemplate(str(path))
    styles = getSampleStyleSheet()
    doc.build([Preformatted(str(text_report), styles["Code"])])
    return path


def export_report(
    report: Any,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
) -> Path:
    """Export a report using a format inferred from the suffix when omitted."""
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".")).lower()
    if fmt == "json":
        return export_json(report, path)
    if fmt == "csv":
        return export_csv(report, path)
    if fmt in {"xlsx", "xls", "excel"}:
        return export_excel(report, path)
    if fmt == "pdf":
        if not isinstance(report, str):
            from agencitylab.api.report import build_text_report

            report = build_text_report(report)
        return export_pdf(report, path)
    raise ValueError(f"Unsupported export format: {fmt}")
