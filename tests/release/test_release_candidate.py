"""Stable-release gates for the public user workflow and critical contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agencitylab import (
    AgencityResult,
    AgencityStream,
    AgencityValidationError,
    PhysicalParameterError,
    __version__,
    analyze_agencity,
    compute_agencity,
    compute_agencity_spectrum,
    export_result_csv,
    export_study_json,
    run_batch,
)
from agencitylab.core.beta import compute_beta

ROOT = Path(__file__).resolve().parents[2]


def _signal(n: int = 801):
    xi = np.linspace(0.0, 20.0, n)
    u = np.sin(xi) + 0.15 * np.sin(2.0 * xi)
    return xi, u


def _compute(xi, u, **kwargs):
    options = dict(A_ref=1.0, tau=2.0, w=1.5, P_c=5.0)
    options.update(kwargs)
    return compute_agencity(u=u, xi=xi, **options)


def test_v1_release_metadata_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert __version__ == "1.1.1"
    assert 'version = "1.1.1"' in pyproject
    assert "Development Status :: 5 - Production/Stable" in pyproject
    assert "Permission is hereby granted" in license_text
    assert "placeholder license text" not in license_text
    assert "version: 1.1.1" in citation
    assert "date-released: 2026-08-12" in citation
    assert "license: MIT" in citation


def test_quickstart_compute_analyze_export_roundtrip(tmp_path):
    xi, u = _signal()
    result = _compute(
        xi,
        u,
        unit="rad",
        coordinate_unit="s",
        power_unit="W",
    )

    assert result.metadata.agencitylab_version == __version__ == "1.1.1"
    assert result.metadata.reference_amplitude == 1.0
    assert result.metadata.characteristic_time == 2.0
    assert result.metadata.memory_window == 1.5
    assert result.metadata.characteristic_power == 5.0
    assert result.coordinate_unit == "s"

    np.testing.assert_allclose(result.S, np.hypot(result.M, result.O))
    expected_beta = np.zeros_like(result.beta)
    mask = result.S > 0.0
    expected_beta[mask] = result.J[mask] * result.U[mask]
    np.testing.assert_allclose(result.beta, expected_beta)
    np.testing.assert_allclose(result.b, result.P_c * result.beta)

    analysis = analyze_agencity(result)
    assert "real_agencity" in analysis
    assert "regime" in analysis

    json_path = export_study_json(result, analysis, tmp_path / "study.json")
    csv_path = export_result_csv(result, tmp_path / "result.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    restored = AgencityResult.from_dict(payload["result"])
    np.testing.assert_allclose(restored.beta, result.beta)
    np.testing.assert_allclose(restored.b, result.b)
    assert restored.metadata.agencitylab_version == "1.1.1"

    frame = pd.read_csv(csv_path)
    for column in ("beta_real", "beta_imag", "b_real", "b_imag"):
        assert column in frame.columns


@pytest.mark.parametrize(
    "bad_signal",
    [
        [],
        [0.0, 1.0],
        [0.0, np.nan, 1.0],
        [0.0, np.inf, 1.0],
        np.zeros((3, 1)),
    ],
)
def test_scalar_api_rejects_invalid_signals_before_numpy_failures(bad_signal):
    with pytest.raises(AgencityValidationError):
        compute_agencity(
            u=bad_signal,
            A_ref=1.0,
            tau=1.0,
            w=1.0,
            P_c=1.0,
        )


def test_scalar_api_rejects_invalid_coordinate_contract():
    u = np.array([0.0, 1.0, 0.0, -1.0])
    with pytest.raises(AgencityValidationError):
        compute_agencity(
            u=u,
            xi=[0.0, 1.0, 1.0, 2.0],
            A_ref=1.0,
            tau=1.0,
            w=1.0,
            P_c=1.0,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("A_ref", 0.0), ("tau", 0.0), ("w", 0.0)],
)
def test_strictly_positive_physical_parameters_reject_zero(parameter, value):
    xi = np.arange(8.0)
    u = np.sin(xi)
    kwargs = dict(A_ref=1.0, tau=1.0, w=1.0, P_c=1.0)
    kwargs[parameter] = value
    with pytest.raises(PhysicalParameterError):
        compute_agencity(u=u, xi=xi, **kwargs)


def test_zero_characteristic_power_is_valid_and_zeroes_flux_only():
    xi = np.arange(16.0)
    result = compute_agencity(
        u=np.sin(0.4 * xi),
        xi=xi,
        A_ref=1.0,
        tau=2.0,
        w=2.0,
        P_c=0.0,
    )
    assert float(result.P_c) == 0.0
    assert result.metadata.characteristic_power == 0.0
    np.testing.assert_array_equal(result.b, 0.0j)
    assert np.any(np.abs(result.beta) > 0.0)


def test_sampled_power_must_match_coordinate():
    xi = np.arange(8.0)
    with pytest.raises(PhysicalParameterError):
        compute_agencity(
            u=np.sin(xi),
            xi=xi,
            A_ref=1.0,
            tau=1.0,
            w=1.0,
            P_c=np.ones(7),
        )


def test_zero_structure_and_critical_surface_follow_canonical_branches():
    D = np.array([0.0, 1.0])
    S = np.array([0.0, 1.0])
    M = np.array([0.0, 1.0])
    O = np.array([0.0, 0.0])
    J, U, beta = compute_beta(D, S, M, O)

    assert U[0] == 0.0j
    assert beta[0] == 0.0j
    assert J[1] == pytest.approx(0.0)
    assert beta[1] == pytest.approx(0.0j)


def test_batch_preserves_order_and_per_item_physics():
    xi = np.arange(64.0)
    u = np.sin(0.2 * xi)
    results = run_batch(
        [
            {"xi": xi, "u": u, "A_ref": 1.0, "tau": 2.0, "w": 2.0, "P_c": 1.0},
            {"xi": xi, "u": u, "A_ref": 2.0, "tau": 3.0, "w": 2.0, "P_c": 4.0},
        ]
    )

    assert [result.A_ref for result in results] == [1.0, 2.0]
    assert [result.tau for result in results] == [2.0, 3.0]
    assert [result.memory_window for result in results] == [2.0, 2.0]
    assert [float(result.P_c) for result in results] == [1.0, 4.0]
    assert all(result.metadata.agencitylab_version == "1.1.1" for result in results)


def test_full_history_stream_matches_one_shot():
    xi = np.arange(64.0)
    u = np.sin(0.2 * xi)
    expected = compute_agencity(
        u=u,
        xi=xi,
        A_ref=1.0,
        tau=2.0,
        w=2.0,
        P_c=1.0,
    )

    stream = AgencityStream(
        analyze=False,
        A_ref=1.0,
        tau=2.0,
        w=2.0,
        P_c=1.0,
    )
    stream.update(u[:32], xi[:32])
    actual = stream.update(u[32:], xi[32:])

    np.testing.assert_allclose(actual.b, expected.b, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.beta, expected.beta, rtol=0.0, atol=0.0)


def test_multiscale_rows_match_independent_scalar_computations():
    xi = np.arange(96.0)
    u = np.sin(0.2 * xi) + 0.1 * np.sin(0.5 * xi)
    taus = [2.0, 4.0]
    windows = [2.0, 3.0]
    spectrum = compute_agencity_spectrum(
        u,
        xi,
        taus,
        A_ref=1.0,
        P_c=2.0,
        windows=windows,
    )

    for index, (tau, w) in enumerate(zip(taus, windows)):
        scalar = compute_agencity(
            u=u,
            xi=xi,
            A_ref=1.0,
            tau=tau,
            w=w,
            P_c=2.0,
        )
        np.testing.assert_allclose(spectrum["b"][index], scalar.b)
