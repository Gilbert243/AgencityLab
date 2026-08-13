import json

import numpy as np
import pandas as pd

from agencitylab import compute_agencity
from agencitylab.api import (
    analyze_agencity,
    export_result_csv,
    export_study_json,
    scientific_workflow,
)


def _result(*, w=None):
    xi = np.linspace(0.0, 20.0, 401)
    u = np.sin(xi) + 0.1 * np.sin(2.0 * xi)
    return compute_agencity(
        u,
        xi,
        A_ref=1.0,
        tau=2.0,
        w=w,
        P_c=3.0,
        unit="m",
        coordinate_unit="s",
        power_unit="W",
    )


def test_explicit_memory_window_is_preserved_by_the_unified_theory_api():
    result = _result(w=1.0)
    assert result.tau == 2.0
    assert result.memory_window == 1.0
    assert result.metadata.extra["memory_window_mode"] == "explicit"

    default = _result()
    assert default.memory_window == default.tau
    assert not np.allclose(result.M, default.M)


def test_structured_analysis_uses_two_crm_windows_for_finite_record_warmup():
    result = _result(w=1.0)
    analysis = analyze_agencity(result)
    interval = analysis["analysis_interval"]
    expected = result.xi[0] + 2.0 * result.memory_window
    assert interval["finite_record_crm_start_time"] >= expected
    assert "2*w" in interval["rule"]


def test_result_csv_is_one_row_per_sample(tmp_path):
    result = _result()
    path = export_result_csv(result, tmp_path / "result.csv")
    frame = pd.read_csv(path)
    assert len(frame) == len(result)
    assert {
        "xi",
        "u",
        "X_star",
        "A_star",
        "M",
        "O",
        "D",
        "S",
        "J",
        "theta",
        "beta_real",
        "beta_imag",
        "b_real",
        "b_imag",
        "b_abs",
        "P_c",
    }.issubset(frame.columns)
    np.testing.assert_allclose(frame["b_abs"], np.abs(result.b))


def test_study_json_preserves_result_and_diagnostics(tmp_path):
    result = _result()
    analysis = analyze_agencity(result)
    path = export_study_json(result, analysis, tmp_path / "study.json", text_report="report")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scientific_ux_schema_version"] == "1.0"
    assert payload["result"]["schema_version"] == "1.0"
    assert payload["analysis"]["real_agencity"]["status"] == "undetermined"
    assert payload["text_report"] == "report"


def test_scientific_workflow_runs_without_figures_or_threshold_inference(tmp_path):
    xi = np.linspace(0.0, 20.0, 401)
    study = scientific_workflow(
        np.sin(xi),
        xi,
        A_ref=1.0,
        tau=2.0,
        w=1.0,
        P_c=2.0,
        figure_kinds=(),
        export_dir=tmp_path,
    )
    assert study.result.memory_window == 1.0
    assert study.analysis["real_agencity"]["status"] == "undetermined"
    assert study.figures == {}
    assert set(study.exports) == {"csv", "json", "report"}
    assert all(path.exists() for path in study.exports.values())
