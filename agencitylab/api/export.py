"""
User-facing export API for AgencityLab.

This module provides convenient export helpers for:
- JSON
- CSV
- Excel
- PDF

The scientific content is expected to come from agencitylab.api.report
or agencitylab.analysis.reports.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from agencitylab.io.json import dump_json
from agencitylab.io.csv import dump_csv


def _ensure_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _flatten_for_table(data: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionaries/lists into a single-level dictionary.
    Useful for CSV and Excel exports.
    """
    if hasattr(data, "to_dict") and callable(data.to_dict):
        data = data.to_dict()
    elif is_dataclass(data):
        data = asdict(data)

    items: Dict[str, Any] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(_flatten_for_table(v, key, sep=sep))
        return items

    if isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten_for_table(v, key, sep=sep))
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
    """
    Export any JSON-serializable Agencity report or result to JSON.
    """
    path = _ensure_path(path)
    return dump_json(data, path, indent=indent, sort_keys=sort_keys)


def export_csv(data: Any, path: Union[str, Path]) -> Path:
    """
    Export a report dictionary to CSV.

    Nested dictionaries are flattened by agencitylab.io.csv.dump_csv.
    """
    path = _ensure_path(path)
    return dump_csv(data, path)


def export_excel(data: Any, path: Union[str, Path]) -> Path:
    """
    Export a report dictionary to Excel (.xlsx).

    Requires pandas.
    """
    path = _ensure_path(path)

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for Excel export") from exc

    flat = _flatten_for_table(data)
    df = pd.DataFrame([flat])
    df.to_excel(path, index=False)
    return path


def export_pdf(text_report: str, path: Union[str, Path]) -> Path:
    """
    Export a text report to PDF.

    Requires reportlab.
    """
    path = _ensure_path(path)

    try:
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Preformatted, SimpleDocTemplate
    except ImportError as exc:
        raise ImportError("reportlab is required for PDF export") from exc

    doc = SimpleDocTemplate(str(path))
    styles = getSampleStyleSheet()

    # Preformatted preserves line breaks and spacing better than Paragraph
    elements = [Preformatted(str(text_report), styles["Code"])]
    doc.build(elements)
    return path


def export_report(
    report: Any,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
) -> Path:
    """
    Export a report using the target format.

    If format is omitted, it is inferred from the file extension.
    """
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt in {"json"}:
        return export_json(report, path)

    if fmt in {"csv"}:
        return export_csv(report, path)

    if fmt in {"xlsx", "xls", "excel"}:
        return export_excel(report, path)

    if fmt in {"pdf"}:
        if not isinstance(report, str):
            # Lazy import to avoid circular dependencies
            from agencitylab.api.report import build_text_report

            report = build_text_report(report)
        return export_pdf(report, path)

    raise ValueError(f"Unsupported export format: {fmt}")