"""Release gates for the current stable AgencityLab software contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import agencitylab as al
from agencitylab.api import (
    AgencityStream,
    analyze_agencity,
    compute_agencity_spectrum,
    export_result_csv,
    export_study_json,
    run_batch,
)
from agencitylab.core.beta import compute_beta

ROOT = Path(__file__).resolve().parents[2]


def _signal(n: int = 801):
    # Half-unit sampling keeps all release-test CRM windows (1.5, 2.0, 3.0)
    # exact integer multiples of the sampling interval. This is a test fixture,
    # not a change to CRM window semantics.
    xi = 0.5 * np.arange(n, dtype=float)
    u = np.sin(0.2 * xi) + 0.15 * np.sin(0.4 * xi)
    return xi, u


def _compute(xi, u, **kwargs):
    options = dict(A_ref=1.0, tau=2.0, w=1.5, P_c=5.0)
    options.update(kwargs)
    return al.compute_agencity(u, xi, **options)


def test_release_metadata_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert al.__version__ == "1.1.0"
    assert 'version = "1.1.0"' in pyproject
    assert "Development Status :: 5 - Production/Stable" in pyproject
    assert "Typing :: Typed" in pyproject
    assert "version: 1.1.0" in citation
    assert "date-released: 2026-08-14" in citation
    assert "license: MIT" in citation
    assert "Permission is hereby granted" in license_text
    assert (ROOT / "agencitylab" / "py.typed").is_file()


def test_quickstart_compute_analyze_export_roundtrip(tmp_path):
    xi, u = _signal()
    result = _compute(xi, u, unit="rad", coordinate_unit="s", power_unit="W")

    assert result.metadata.agencitylab_version == al.__version__ == "1.1.0"
    np.testing.assert_allclose(result.S, np.hypot(result.M, result.O))
    expected_beta = np.zeros_like(result.beta)
    mask = result.S > 0.0
    expected_beta[mask] = result.J[mask] * result.U[mask]
    np.testing.assert_allclose(result.beta, expected_beta)
    np.testing.assert_allclose(result.b, result.P_c * result.beta)

    analysis = analyze_agencity(result)
    json_path = export_study_json(result, analysis, tmp_path / "study.json")
    csv_path = export_result_csv(result, tmp_path / "result.csv")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["scientific_ux_schema_version"] == "1.0"
    restored = al.AgencityResult.from_dict(payload["result"])
    np.testing.assert_allclose(restored.beta, result.beta)
    np.testing.assert_allclose(restored.b, result.b)
    frame = pd.read_csv(csv_path)
    for column in ("beta_real", "beta_imag", "b_real", "b_imag"):
        assert column in frame.columns


def test_zero_structure_and_zero_power_keep_exact_canonical_branches():
    D = np.array([0.0, 1.0])
    S = np.array([0.0, 1.0])
    M = np.array([0.0, 1.0])
    O = np.array([0.0, 0.0])
    J, U, beta = compute_beta(D, S, M, O)
    assert U[0] == 0.0j
    assert beta[0] == 0.0j
    assert J[1] == 0.0

    xi, u = _signal(64)
    result = al.compute_agencity(u, xi, A_ref=1.0, tau=2.0, w=2.0, P_c=0.0)
    np.testing.assert_array_equal(result.b, 0.0j)


def test_batch_streaming_and_multiscale_remain_equivalent_to_scalar_compute():
    xi, u = _signal(96)
    batch = run_batch(
        [
            {"xi": xi, "u": u, "A_ref": 1.0, "tau": 2.0, "w": 2.0, "P_c": 1.0},
            {"xi": xi, "u": u, "A_ref": 1.0, "tau": 4.0, "w": 3.0, "P_c": 2.0},
        ]
    )
    assert [float(result.P_c) for result in batch] == [1.0, 2.0]

    expected = al.compute_agencity(u, xi, A_ref=1.0, tau=2.0, w=2.0, P_c=1.0)
    stream = AgencityStream(analyze=False, A_ref=1.0, tau=2.0, w=2.0, P_c=1.0)
    stream.update(u[:48], xi[:48])
    actual = stream.update(u[48:], xi[48:])
    np.testing.assert_array_equal(actual.beta, expected.beta)
    np.testing.assert_array_equal(actual.b, expected.b)

    spectrum = compute_agencity_spectrum(
        u, xi, [2.0, 4.0], A_ref=1.0, P_c=2.0, windows=[2.0, 3.0]
    )
    for index, (tau, w) in enumerate(((2.0, 2.0), (4.0, 3.0))):
        scalar = al.compute_agencity(u, xi, A_ref=1.0, tau=tau, w=w, P_c=2.0)
        np.testing.assert_allclose(spectrum["b"][index], scalar.b)
